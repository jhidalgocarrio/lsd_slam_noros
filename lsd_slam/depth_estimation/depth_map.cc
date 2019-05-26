/**
* This file is part of LSD-SLAM.
*
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam>
*
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "depth_estimation/depth_map.h"

#include <chrono>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "util/settings.h"
#include "depth_estimation/depth_map_pixel_hypothesis.h"
#include "model/frame.h"
#include "util/global_funcs.h"
#include "io_wrapper/image_display.h"
#include "global_mapping/key_frame_graph.h"
#include "projection/projection.h"

namespace lsd_slam
{

DepthMap::DepthMap(int w, int h, const Eigen::Matrix3f& K)
{
    width = w;
    height = h;

    activeKeyFrame = 0;
    activeKeyFrameIsReactivated = false;
    otherDepthMap = new DepthMapPixelHypothesis[width*height];
    currentDepthMap = new DepthMapPixelHypothesis[width*height];

    validityIntegralBuffer = new int[width*height];

    debugImageHypothesisHandling = cv::Mat(h,w, CV_8UC3);
    debugImageHypothesisPropagation = cv::Mat(h,w, CV_8UC3);
    debugImageStereoLines = cv::Mat(h,w, CV_8UC3);
    debugImageDepth = cv::Mat(h,w, CV_8UC3);

    this->K = K;
    focal_length << K(0,0), K(1,1);
    offset = K.block(0, 2, 2, 1);

    KInv = K.inverse();
    fxi = KInv(0,0);
    fyi = KInv(1,1);
    cxi = KInv(0,2);
    cyi = KInv(1,2);

    reset();

    msUpdate = msCreate = msFinalize = msObserve
             = msRegularize = msPropagate = msFillHoles = msSetDepth = 0;
    lastHzUpdate = std::chrono::high_resolution_clock::now();
    nUpdate = nCreate = nFinalize = nObserve
            = nRegularize = nPropagate = nFillHoles = nSetDepth = 0;
    nAvgUpdate = nAvgCreate = nAvgFinalize = nAvgObserve = nAvgRegularize
               = nAvgPropagate = nAvgFillHoles = nAvgSetDepth = 0;
}

DepthMap::~DepthMap()
{
    if(activeKeyFrame != 0)
        activeKeyFramelock.unlock();

    debugImageHypothesisHandling.release();
    debugImageHypothesisPropagation.release();
    debugImageStereoLines.release();
    debugImageDepth.release();

    delete[] otherDepthMap;
    delete[] currentDepthMap;

    delete[] validityIntegralBuffer;
}


void DepthMap::reset()
{
    for(DepthMapPixelHypothesis* pt = otherDepthMap+width*height-1; pt >= otherDepthMap; pt--)
        pt->isValid = false;
    for(DepthMapPixelHypothesis* pt = currentDepthMap+width*height-1; pt >= currentDepthMap; pt--)
        pt->isValid = false;
}


void DepthMap::observeDepthRow(int yMin, int yMax, RunningStats* stats)
{
    const float* keyFrameMaxGradBuf = activeKeyFrame->maxGradients(0);

    int successes = 0;

    for(int y=yMin; y<yMax; y++)
        for(int x=3; x<width-3; x++)
        {
            int idx = x+y*width;
            DepthMapPixelHypothesis* target = currentDepthMap+idx;
            bool hasHypothesis = target->isValid;

            // ======== 1. check absolute grad =========
            if(hasHypothesis && keyFrameMaxGradBuf[idx] < MIN_ABS_GRAD_DECREASE)
            {
                target->isValid = false;
                continue;
            }

            if(keyFrameMaxGradBuf[idx] < MIN_ABS_GRAD_CREATE
                    || target->blacklisted < MIN_BLACKLIST)
                continue;

            Eigen::Vector2i p;
            p << x, y;

            bool success;
            if(!hasHypothesis)
                success = observeDepthCreate(p, idx, stats);
            else
                success = observeDepthUpdate(p, idx, keyFrameMaxGradBuf, stats);

            if(success)
                successes++;
        }


}
void DepthMap::observeDepth()
{

    threadReducer.reduce(boost::bind(&DepthMap::observeDepthRow, this, _1, _2, _3),
                         3, height-3, 10);

    if(enablePrintDebugInfo && printObserveStatistics)
    {
        printf("OBSERVE (%d): %d / %d created; %d / %d updated; %d skipped; %d init-blacklisted\n",
               activeKeyFrame->id(),
               runningStats.num_observe_created,
               runningStats.num_observe_create_attempted,
               runningStats.num_observe_updated,
               runningStats.num_observe_update_attempted,
               runningStats.num_observe_skip_alreadyGood,
               runningStats.num_observe_blacklisted
              );
    }


    if(enablePrintDebugInfo && printObservePurgeStatistics)
    {
        printf("OBS-PRG (%d): Good: %d; inconsistent: %d; notfound: %d; oob: %d; failed: %d; addSkip: %d;\n",
               activeKeyFrame->id(),
               runningStats.num_observe_good,
               runningStats.num_observe_inconsistent,
               runningStats.num_observe_notfound,
               runningStats.num_observe_skip_oob,
               runningStats.num_observe_skip_fail,
               runningStats.num_observe_addSkip
              );
    }
}


bool DepthMap::makeAndCheckEPL(const Eigen::Vector2f &p, const Frame* const ref,
                               Eigen::Vector2f &pep, RunningStats* const stats)
{
    int idx = p(0) + p(1)*width;

    // ======= make epl ========
    // calculate the plane spanned by the two camera centers and the point (p(0),p(1),1)
    // intersect it with the keyframe's image plane (at depth=1)
    Eigen::Vector2f epn = - focal_length.cwiseProduct(ref->thisToOther_t.segment(0, 2))
                        + ref->thisToOther_t[2]*(p - offset);

    if(std::isnan(epn.sum()))
        return false;


    // ======== check epl length =========
    float eplLengthSquared = epn.dot(epn);
    if(eplLengthSquared < MIN_EPL_LENGTH_SQUARED)
    {
        if(enablePrintDebugInfo) stats->num_observe_skipped_small_epl++;
        return false;
    }


    // ===== check epl-grad magnitude ======
    float gx = activeKeyFrameImageData[idx+1] - activeKeyFrameImageData[idx-1];
    float gy = activeKeyFrameImageData[idx+width] - activeKeyFrameImageData[idx-width];
    Eigen::Vector2f grad;
    grad << gx, gy;
    float eplGradSquared = grad.dot(epn);
    eplGradSquared = eplGradSquared*eplGradSquared /
                     eplLengthSquared;	// square and norm with epl-length

    if(eplGradSquared < MIN_EPL_GRAD_SQUARED)
    {
        if(enablePrintDebugInfo) stats->num_observe_skipped_small_epl_grad++;
        return false;
    }


    // ===== check epl-grad angle ======
    if(eplGradSquared / grad.dot(grad) < MIN_EPL_ANGLE_SQUARED)
    {
        if(enablePrintDebugInfo) stats->num_observe_skipped_small_epl_angle++;
        return false;
    }


    // ===== DONE - return "normalized" epl =====
    float fac = GRADIENT_SAMPLE_DIST / sqrt(eplLengthSquared);
    pep = epn * fac;

    return true;
}


bool DepthMap::observeDepthCreate(const Eigen::Vector2i &p, const int &idx,
                                  RunningStats* const &stats)
{
    DepthMapPixelHypothesis* target = currentDepthMap+idx;

    Frame* refFrame = activeKeyFrameIsReactivated ? newest_referenceFrame :
                      oldest_referenceFrame;

    if(refFrame->getTrackingParent() == activeKeyFrame)
    {
        bool* wasGoodDuringTracking = refFrame->refPixelWasGoodNoCreate();
        if(wasGoodDuringTracking != 0 &&
           !wasGoodDuringTracking[
            (p(0) >> SE3TRACKING_MIN_LEVEL) +
            (width >> SE3TRACKING_MIN_LEVEL)*(p(1) >> SE3TRACKING_MIN_LEVEL)
           ])
        {
            if(plotStereoImages) {
                // BLUE for SKIPPED NOT GOOD TRACKED
                debugImageHypothesisHandling.at<cv::Vec3b>(p(1), p(0)) = cv::Vec3b(255,0,0);
            }
            return false;
        }
    }

    Eigen::Vector2f epn;
    bool isGood = makeAndCheckEPL(p.cast<float>(), refFrame, epn, stats);
    if(!isGood) return false;

    if(enablePrintDebugInfo) stats->num_observe_create_attempted++;

    float result_idepth, result_var, result_eplLength;

    float error = doLineStereo(
                      p.cast<float>(), epn,
                      0.0f, 1.0f, 1.0f/MIN_DEPTH,
                      refFrame, refFrame->image(0),
                      result_idepth, result_var, result_eplLength, stats);

    if(error == -3 || error == -2)
    {
        target->blacklisted--;
        if(enablePrintDebugInfo) stats->num_observe_blacklisted++;
    }

    if(error < 0 || result_var > MAX_VAR)
        return false;

    result_idepth = UNZERO(result_idepth);

    // add hypothesis
    *target = DepthMapPixelHypothesis(
                  result_idepth,
                  result_var,
                  VALIDITY_COUNTER_INITIAL_OBSERVE);

    if(plotStereoImages)
        debugImageHypothesisHandling.at<cv::Vec3b>(p(1), p(0)) = cv::Vec3b(255,255,255); // white for GOT CREATED

    if(enablePrintDebugInfo) stats->num_observe_created++;

    return true;
}

bool DepthMap::observeDepthUpdate(const Eigen::Vector2i &p, const int &idx,
                                  const float* keyFrameMaxGradBuf, RunningStats* const &stats)
{
    DepthMapPixelHypothesis* target = currentDepthMap+idx;
    Frame* refFrame;


    if(!activeKeyFrameIsReactivated)
    {
        if((int)target->nextStereoFrameMinID - referenceFrameByID_offset >=
                (int)referenceFrameByID.size())
        {
            if(plotStereoImages)
                debugImageHypothesisHandling.at<cv::Vec3b>(p(1), p(0)) = cv::Vec3b(0,255,
                        0);	// GREEN FOR skip

            if(enablePrintDebugInfo) stats->num_observe_skip_alreadyGood++;
            return false;
        }

        if((int)target->nextStereoFrameMinID - referenceFrameByID_offset < 0)
            refFrame = oldest_referenceFrame;
        else
            refFrame = referenceFrameByID[(int)target->nextStereoFrameMinID - referenceFrameByID_offset];
    }
    else
        refFrame = newest_referenceFrame;


    if(refFrame->getTrackingParent() == activeKeyFrame)
    {
        bool* wasGoodDuringTracking = refFrame->refPixelWasGoodNoCreate();
        if(wasGoodDuringTracking != 0 &&
           !wasGoodDuringTracking[
               (p(0) >> SE3TRACKING_MIN_LEVEL) +
               (width >> SE3TRACKING_MIN_LEVEL)*(p(1) >> SE3TRACKING_MIN_LEVEL)
           ])
        {
            if(plotStereoImages)
                debugImageHypothesisHandling.at<cv::Vec3b>(p(1), p(0)) = cv::Vec3b(255,0,
                        0); // BLUE for SKIPPED NOT GOOD TRACKED
            return false;
        }
    }

    Eigen::Vector2f epn;
    bool isGood = makeAndCheckEPL(p.cast<float>(), refFrame, epn, stats);
    if(!isGood) return false;

    // which exact point to track, and where from.
    float sv = sqrt(target->idepth_var_smoothed);
    float min_idepth = target->idepth_smoothed - sv*STEREO_EPL_VAR_FAC;
    float max_idepth = target->idepth_smoothed + sv*STEREO_EPL_VAR_FAC;
    if(min_idepth < 0) min_idepth = 0;
    if(max_idepth > 1/MIN_DEPTH) max_idepth = 1/MIN_DEPTH;

    stats->num_observe_update_attempted++;

    float result_idepth, result_var, result_eplLength;

    float error = doLineStereo(
                      p.cast<float>(), epn,
                      min_idepth, target->idepth_smoothed,max_idepth,
                      refFrame, refFrame->image(0),
                      result_idepth, result_var, result_eplLength, stats);

    float diff = result_idepth - target->idepth_smoothed;


    // if oob: (really out of bounds)
    if(error == -1)
    {
        // do nothing, pixel got oob, but is still in bounds in original. I will want to try again.
        if(enablePrintDebugInfo) stats->num_observe_skip_oob++;

        if(plotStereoImages) {
            // RED FOR OOB
            debugImageHypothesisHandling.at<cv::Vec3b>(p(1), p(0)) = cv::Vec3b(0,0,255);
        }
        return false;
    }

    // if just not good for stereo (e.g. some inf / nan occured; has inconsistent minimum; ..)
    else if(error == -2)
    {
        if(enablePrintDebugInfo) stats->num_observe_skip_fail++;

        if(plotStereoImages) {
            // PURPLE FOR NON-GOOD
            debugImageHypothesisHandling.at<cv::Vec3b>(p(1), p(0)) = cv::Vec3b(255,0,255);
        }


        target->validity_counter -= VALIDITY_COUNTER_DEC;
        if(target->validity_counter < 0) target->validity_counter = 0;


        target->nextStereoFrameMinID = 0;

        target->idepth_var *= FAIL_VAR_INC_FAC;
        if(target->idepth_var > MAX_VAR)
        {
            target->isValid = false;
            target->blacklisted--;
        }
        return false;
    }

    // if not found (error too high)
    else if(error == -3)
    {
        if(enablePrintDebugInfo) stats->num_observe_notfound++;
        if(plotStereoImages) {
            // BLACK FOR big not-found
            debugImageHypothesisHandling.at<cv::Vec3b>(p(1), p(0)) = cv::Vec3b(0,0,0);
        }


        return false;
    }

    else if(error == -4)
    {
        if(plotStereoImages) {
            // BLACK FOR big arithmetic error
            debugImageHypothesisHandling.at<cv::Vec3b>(p(1), p(0)) = cv::Vec3b(0,0,0);
        }

        return false;
    }

    // if inconsistent
    else if(DIFF_FAC_OBSERVE*diff*diff > result_var + target->idepth_var_smoothed)
    {
        if(enablePrintDebugInfo) stats->num_observe_inconsistent++;
        if(plotStereoImages) {
            // Turkoise FOR big inconsistent
            debugImageHypothesisHandling.at<cv::Vec3b>(p(1), p(0)) = cv::Vec3b(255,255,0);
        }

        target->idepth_var *= FAIL_VAR_INC_FAC;
        if(target->idepth_var > MAX_VAR) target->isValid = false;

        return false;
    }


    else
    {
        // one more successful observation!
        if(enablePrintDebugInfo) stats->num_observe_good++;

        if(enablePrintDebugInfo) stats->num_observe_updated++;


        // do textbook ekf update:
        // increase var by a little (prediction-uncertainty)
        float id_var = target->idepth_var*SUCC_VAR_INC_FAC;

        // update var with observation
        float w = result_var / (result_var + id_var);
        float new_idepth = (1-w)*result_idepth + w*target->idepth;
        target->idepth = UNZERO(new_idepth);

        // variance can only decrease from observation; never increase.
        id_var = id_var * w;
        if(id_var < target->idepth_var)
            target->idepth_var = id_var;

        // increase validity!
        target->validity_counter += VALIDITY_COUNTER_INC;
        float absGrad = keyFrameMaxGradBuf[idx];
        if(target->validity_counter > VALIDITY_COUNTER_MAX+absGrad*
                (VALIDITY_COUNTER_MAX_VARIABLE)/255.0f)
            target->validity_counter = VALIDITY_COUNTER_MAX+absGrad*
                                       (VALIDITY_COUNTER_MAX_VARIABLE)/255.0f;

        // increase Skip!
        if(result_eplLength < MIN_EPL_LENGTH_CROP)
        {
            float inc = activeKeyFrame->numFramesTrackedOnThis / (float)(
                            activeKeyFrame->numMappedOnThis+5);
            if(inc < 3) inc = 3;

            inc +=  ((int)(result_eplLength*10000)%2);

            if(enablePrintDebugInfo) stats->num_observe_addSkip++;

            if(result_eplLength < 0.5*MIN_EPL_LENGTH_CROP)
                inc *= 3;


            target->nextStereoFrameMinID = refFrame->id() + inc;
        }

        if(plotStereoImages) {
            // yellow for GOT UPDATED
            debugImageHypothesisHandling.at<cv::Vec3b>(p(1), p(0)) = cv::Vec3b(0,255,255);
        }
        return true;
    }
}

void DepthMap::propagateDepth(Frame* new_keyframe)
{
    runningStats.num_prop_removed_out_of_bounds = 0;
    runningStats.num_prop_removed_colorDiff = 0;
    runningStats.num_prop_removed_validity = 0;
    runningStats.num_prop_grad_decreased = 0;
    runningStats.num_prop_color_decreased = 0;
    runningStats.num_prop_attempts = 0;
    runningStats.num_prop_occluded = 0;
    runningStats.num_prop_created = 0;
    runningStats.num_prop_merged = 0;


    if(new_keyframe->getTrackingParent() != activeKeyFrame)
    {
        printf("WARNING: propagating depth from frame %d to %d, which was tracked on a different frame (%d).\nWhile this should work, it is not recommended.",
               activeKeyFrame->id(), new_keyframe->id(),
               new_keyframe->getTrackingParent()->id());
    }

    // wipe depthmap
    for(DepthMapPixelHypothesis* pt = otherDepthMap+width*height-1;
            pt >= otherDepthMap; pt--)
    {
        pt->isValid = false;
        pt->blacklisted = 0;
    }

    // re-usable values.
    SE3 oldToNew_SE3 = se3FromSim3(new_keyframe->pose->thisToParent_raw).inverse();
    Eigen::Vector3f trafoInv_t = oldToNew_SE3.translation().cast<float>();
    Eigen::Matrix3f trafoInv_R =
        oldToNew_SE3.rotationMatrix().matrix().cast<float>();


    const bool* trackingWasGood = new_keyframe->getTrackingParent() ==
                                  activeKeyFrame ? new_keyframe->refPixelWasGoodNoCreate() : 0;


    const float* activeKFImageData = activeKeyFrame->image(0);
    const float* newKFMaxGrad = new_keyframe->maxGradients(0);
    const float* newKFImageData = new_keyframe->image(0);





    // go through all pixels of OLD image, propagating forwards.
    for(int y=0; y<height; y++)
        for(int x=0; x<width; x++)
        {
            DepthMapPixelHypothesis* source = currentDepthMap + x + y*width;

            if(!source->isValid)
                continue;

            if(enablePrintDebugInfo) runningStats.num_prop_attempts++;


            Eigen::Vector3f pn = (trafoInv_R * Eigen::Vector3f(x*fxi + cxi,y*fyi + cyi,
                                  1.0f)) / source->idepth_smoothed + trafoInv_t;

            float new_idepth = 1.0f / pn[2];

            Eigen::Vector2f p = new_idepth * focal_length.cwiseProduct(pn.segment(0, 2)) + offset;

            // check if still within image, if not: DROP.
            if(!(p(0) > 2.1f && p(1) > 2.1f && p(0) < width-3.1f
                    && p(1) < height-3.1f))
            {
                if(enablePrintDebugInfo) runningStats.num_prop_removed_out_of_bounds++;
                continue;
            }

            int newIDX = (int)(p(0)+0.5f) + ((int)(p(1)+0.5f))*width;
            float destAbsGrad = newKFMaxGrad[newIDX];

            if(trackingWasGood != 0)
            {
                if(!trackingWasGood[(x >> SE3TRACKING_MIN_LEVEL) + (width >>
                                                                 SE3TRACKING_MIN_LEVEL)*(y >> SE3TRACKING_MIN_LEVEL)]
                        || destAbsGrad < MIN_ABS_GRAD_DECREASE)
                {
                    if(enablePrintDebugInfo) runningStats.num_prop_removed_colorDiff++;
                    continue;
                }
            }
            else
            {
                float sourceColor = activeKFImageData[x + y*width];
                float destColor = getInterpolatedElement(newKFImageData, p, width);

                float residual = destColor - sourceColor;


                if(residual*residual / (MAX_DIFF_CONSTANT +
                                        MAX_DIFF_GRAD_MULT*destAbsGrad*destAbsGrad) > 1
                        || destAbsGrad < MIN_ABS_GRAD_DECREASE)
                {
                    if(enablePrintDebugInfo) runningStats.num_prop_removed_colorDiff++;
                    continue;
                }
            }

            DepthMapPixelHypothesis* targetBest = otherDepthMap +  newIDX;

            // large idepth = point is near = large increase in variance.
            // small idepth = point is far = small increase in variance.
            float idepth_ratio_4 = new_idepth / source->idepth_smoothed;
            idepth_ratio_4 *= idepth_ratio_4;
            idepth_ratio_4 *= idepth_ratio_4;

            float new_var =idepth_ratio_4*source->idepth_var;


            // check for occlusion
            if(targetBest->isValid)
            {
                // if they occlude one another, one gets removed.
                float diff = targetBest->idepth - new_idepth;
                if(DIFF_FAC_PROP_MERGE*diff*diff >
                        new_var +
                        targetBest->idepth_var)
                {
                    if(new_idepth < targetBest->idepth)
                    {
                        if(enablePrintDebugInfo) runningStats.num_prop_occluded++;
                        continue;
                    }
                    else
                    {
                        if(enablePrintDebugInfo) runningStats.num_prop_occluded++;
                        targetBest->isValid = false;
                    }
                }
            }


            if(!targetBest->isValid)
            {
                if(enablePrintDebugInfo) runningStats.num_prop_created++;

                *targetBest = DepthMapPixelHypothesis(
                                  new_idepth,
                                  new_var,
                                  source->validity_counter);

            }
            else
            {
                if(enablePrintDebugInfo) runningStats.num_prop_merged++;

                // merge idepth ekf-style
                float w = new_var / (targetBest->idepth_var + new_var);
                float merged_new_idepth = w*targetBest->idepth + (1-w)*new_idepth;

                // merge validity
                int merged_validity = source->validity_counter + targetBest->validity_counter;
                if(merged_validity > VALIDITY_COUNTER_MAX+(VALIDITY_COUNTER_MAX_VARIABLE))
                    merged_validity = VALIDITY_COUNTER_MAX+(VALIDITY_COUNTER_MAX_VARIABLE);

                *targetBest = DepthMapPixelHypothesis(
                                  merged_new_idepth,
                                  1.0f/(1.0f/targetBest->idepth_var + 1.0f/new_var),
                                  merged_validity);
            }
        }

    // swap!
    std::swap(currentDepthMap, otherDepthMap);


    if(enablePrintDebugInfo && printPropagationStatistics)
    {
        printf("PROPAGATE: %d: %d drop (%d oob, %d color); %d created; %d merged; %d occluded. %d col-dec, %d grad-dec.\n",
               runningStats.num_prop_attempts,
               runningStats.num_prop_removed_validity +
               runningStats.num_prop_removed_out_of_bounds +
               runningStats.num_prop_removed_colorDiff,
               runningStats.num_prop_removed_out_of_bounds,
               runningStats.num_prop_removed_colorDiff,
               runningStats.num_prop_created,
               runningStats.num_prop_merged,
               runningStats.num_prop_occluded,
               runningStats.num_prop_color_decreased,
               runningStats.num_prop_grad_decreased);
    }
}


void DepthMap::regularizeDepthMapFillHolesRow(int yMin, int yMax,
        RunningStats* stats)
{
    // =========== regularize fill holes
    const float* keyFrameMaxGradBuf = activeKeyFrame->maxGradients(0);

    for(int y=yMin; y<yMax; y++)
    {
        for(int x=3; x<width-2; x++)
        {
            int idx = x+y*width;
            DepthMapPixelHypothesis* dest = otherDepthMap + idx;
            if(dest->isValid) continue;
            if(keyFrameMaxGradBuf[idx]<MIN_ABS_GRAD_DECREASE) continue;

            int* io = validityIntegralBuffer + idx;
            int val = io[2+2*width] - io[2-3*width] - io[-3+2*width] + io[-3-3*width];


            if((dest->blacklisted >= MIN_BLACKLIST && val > VAL_SUM_MIN_FOR_CREATE)
                    || val > VAL_SUM_MIN_FOR_UNBLACKLIST)
            {
                float sumIdepthObs = 0, sumIVarObs = 0;
                int num = 0;

                DepthMapPixelHypothesis* s1max = otherDepthMap + (x-2) + (y+3)*width;
                for (DepthMapPixelHypothesis* s1 = otherDepthMap + (x-2) + (y-2)*width;
                        s1 < s1max; s1+=width)
                    for(DepthMapPixelHypothesis* source = s1; source < s1+5; source++)
                    {
                        if(!source->isValid) continue;

                        sumIdepthObs += source->idepth /source->idepth_var;
                        sumIVarObs += 1.0f/source->idepth_var;
                        num++;
                    }

                float idepthObs = sumIdepthObs / sumIVarObs;
                idepthObs = UNZERO(idepthObs);

                currentDepthMap[idx] =
                    DepthMapPixelHypothesis(
                        idepthObs,
                        VAR_RANDOM_INIT_INITIAL,
                        0);

                if(enablePrintDebugInfo) stats->num_reg_created++;
            }
        }
    }
}


void DepthMap::regularizeDepthMapFillHoles()
{

    buildRegIntegralBuffer();

    runningStats.num_reg_created=0;

    memcpy(otherDepthMap,currentDepthMap,
           width*height*sizeof(DepthMapPixelHypothesis));
    threadReducer.reduce(boost::bind(&DepthMap::regularizeDepthMapFillHolesRow,
                                     this, _1, _2, _3), 3, height-2, 10);
    if(enablePrintDebugInfo && printFillHolesStatistics)
        printf("FillHoles (discreteDepth): %d created\n",
               runningStats.num_reg_created);
}



void DepthMap::buildRegIntegralBufferRow1(int yMin, int yMax,
        RunningStats* stats)
{
    // ============ build inegral buffers
    int* validityIntegralBufferPT = validityIntegralBuffer+yMin*width;
    DepthMapPixelHypothesis* ptSrc = currentDepthMap+yMin*width;
    for(int y=yMin; y<yMax; y++)
    {
        int validityIntegralBufferSUM = 0;

        for(int x=0; x<width; x++)
        {
            if(ptSrc->isValid)
                validityIntegralBufferSUM += ptSrc->validity_counter;

            *(validityIntegralBufferPT++) = validityIntegralBufferSUM;
            ptSrc++;
        }
    }
}


void DepthMap::buildRegIntegralBuffer()
{
    threadReducer.reduce(boost::bind(&DepthMap::buildRegIntegralBufferRow1, this,
                                     _1, _2,_3), 0, height);

    int* validityIntegralBufferPT = validityIntegralBuffer;
    int* validityIntegralBufferPT_T = validityIntegralBuffer+width;

    int wh = height*width;
    for(int idx=width; idx<wh; idx++)
        *(validityIntegralBufferPT_T++) += *(validityIntegralBufferPT++);

}



template<bool removeOcclusions> void DepthMap::regularizeDepthMapRow(
    int validityTH, int yMin, int yMax, RunningStats* stats)
{
    const int regularize_radius = 2;

    const float regDistVar = REG_DIST_VAR;

    for(int y=yMin; y<yMax; y++)
    {
        for(int x=regularize_radius; x<width-regularize_radius; x++)
        {
            DepthMapPixelHypothesis* dest = currentDepthMap + x + y*width;
            DepthMapPixelHypothesis* destRead = otherDepthMap + x + y*width;

            // if isValid need to do better examination and then update.

            if(enablePrintDebugInfo && destRead->blacklisted < MIN_BLACKLIST)
                stats->num_reg_blacklisted++;

            if(!destRead->isValid)
                continue;

            float sum=0, val_sum=0, sumIvar=0;//, min_varObs = 1e20;
            int numOccluding = 0, numNotOccluding = 0;

            for(int dx=-regularize_radius; dx<=regularize_radius; dx++)
                for(int dy=-regularize_radius; dy<=regularize_radius; dy++)
                {
                    DepthMapPixelHypothesis* source = destRead + dx + dy*width;

                    if(!source->isValid) continue;
//					stats->num_reg_total++;

                    float diff =source->idepth - destRead->idepth;
                    if(DIFF_FAC_SMOOTHING*diff*diff > source->idepth_var + destRead->idepth_var)
                    {
                        if(removeOcclusions)
                        {
                            if(source->idepth > destRead->idepth)
                                numOccluding++;
                        }
                        continue;
                    }

                    val_sum += source->validity_counter;

                    if(removeOcclusions)
                        numNotOccluding++;

                    float distFac = (float)(dx*dx+dy*dy)*regDistVar;
                    float ivar = 1.0f/(source->idepth_var + distFac);

                    sum += source->idepth * ivar;
                    sumIvar += ivar;


                }

            if(val_sum < validityTH)
            {
                dest->isValid = false;
                if(enablePrintDebugInfo) stats->num_reg_deleted_secondary++;
                dest->blacklisted--;

                if(enablePrintDebugInfo) stats->num_reg_setBlacklisted++;
                continue;
            }


            if(removeOcclusions)
            {
                if(numOccluding > numNotOccluding)
                {
                    dest->isValid = false;
                    if(enablePrintDebugInfo) stats->num_reg_deleted_occluded++;

                    continue;
                }
            }

            sum = sum / sumIvar;
            sum = UNZERO(sum);


            // update!
            dest->idepth_smoothed = sum;
            dest->idepth_var_smoothed = 1.0f/sumIvar;

            if(enablePrintDebugInfo) stats->num_reg_smeared++;
        }
    }
}
template void DepthMap::regularizeDepthMapRow<true>(int validityTH, int yMin,
        int yMax, RunningStats* stats);
template void DepthMap::regularizeDepthMapRow<false>(int validityTH, int yMin,
        int yMax, RunningStats* stats);


void DepthMap::regularizeDepthMap(bool removeOcclusions, int validityTH)
{
    runningStats.num_reg_smeared=0;
    runningStats.num_reg_total=0;
    runningStats.num_reg_deleted_secondary=0;
    runningStats.num_reg_deleted_occluded=0;
    runningStats.num_reg_blacklisted=0;
    runningStats.num_reg_setBlacklisted=0;

    memcpy(otherDepthMap,currentDepthMap,
           width*height*sizeof(DepthMapPixelHypothesis));


    if(removeOcclusions)
        threadReducer.reduce(boost::bind(&DepthMap::regularizeDepthMapRow<true>, this,
                                         validityTH, _1, _2, _3), 2, height-2, 10);
    else
        threadReducer.reduce(boost::bind(&DepthMap::regularizeDepthMapRow<false>, this,
                                         validityTH, _1, _2, _3), 2, height-2, 10);


    if(enablePrintDebugInfo && printRegularizeStatistics)
        printf("REGULARIZE (%d): %d smeared; %d blacklisted /%d new); %d deleted; %d occluded; %d filled\n",
               activeKeyFrame->id(),
               runningStats.num_reg_smeared,
               runningStats.num_reg_blacklisted,
               runningStats.num_reg_setBlacklisted,
               runningStats.num_reg_deleted_secondary,
               runningStats.num_reg_deleted_occluded,
               runningStats.num_reg_created);
}


void DepthMap::initializeRandomly(Frame* new_frame)
{
    activeKeyFramelock = new_frame->getActiveLock();
    activeKeyFrame = new_frame;
    activeKeyFrameImageData = activeKeyFrame->image(0);
    activeKeyFrameIsReactivated = false;

    const float* maxGradients = new_frame->maxGradients();

    for(int y=1; y<height-1; y++)
    {
        for(int x=1; x<width-1; x++)
        {
            if(maxGradients[x+y*width] > MIN_ABS_GRAD_CREATE)
            {
                float idepth = 0.5f + 1.0f * ((rand() % 100001) / 100000.0f);
                currentDepthMap[x+y*width] = DepthMapPixelHypothesis(
                                                 idepth,
                                                 idepth,
                                                 VAR_RANDOM_INIT_INITIAL,
                                                 VAR_RANDOM_INIT_INITIAL,
                                                 20);
            }
            else
            {
                currentDepthMap[x+y*width].isValid = false;
                currentDepthMap[x+y*width].blacklisted = 0;
            }
        }
    }


    activeKeyFrame->setDepth(currentDepthMap);
}



void DepthMap::setFromExistingKF(Frame* kf)
{
    assert(kf->hasIDepthBeenSet());

    activeKeyFramelock = kf->getActiveLock();
    activeKeyFrame = kf;

    const float* idepth = activeKeyFrame->idepth_reAct();
    const float* idepthVar = activeKeyFrame->idepthVar_reAct();
    const unsigned char* validity = activeKeyFrame->validity_reAct();

    DepthMapPixelHypothesis* pt = currentDepthMap;
    activeKeyFrame->numMappedOnThis = 0;
    activeKeyFrame->numFramesTrackedOnThis = 0;
    activeKeyFrameImageData = activeKeyFrame->image(0);
    activeKeyFrameIsReactivated = true;

    for(int y=0; y<height; y++)
    {
        for(int x=0; x<width; x++)
        {
            if(*idepthVar > 0)
            {
                *pt = DepthMapPixelHypothesis(
                          *idepth,
                          *idepthVar,
                          *validity);
            }
            else
            {
                currentDepthMap[x+y*width].isValid = false;
                currentDepthMap[x+y*width].blacklisted = (*idepthVar == -2) ? MIN_BLACKLIST-1 :
                        0;
            }

            idepth++;
            idepthVar++;
            validity++;
            pt++;
        }
    }

    regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
}


void DepthMap::initializeFromGTDepth(Frame* new_frame)
{
    assert(new_frame->hasIDepthBeenSet());

    activeKeyFramelock = new_frame->getActiveLock();
    activeKeyFrame = new_frame;
    activeKeyFrameImageData = activeKeyFrame->image(0);
    activeKeyFrameIsReactivated = false;

    const float* idepth = new_frame->idepth();


    float averageGTIDepthSum = 0;
    int averageGTIDepthNum = 0;
    for(int y=0; y<height; y++)
    {
        for(int x=0; x<width; x++)
        {
            float idepthValue = idepth[x+y*width];
            if(!std::isnan(idepthValue) && idepthValue > 0)
            {
                averageGTIDepthSum += idepthValue;
                averageGTIDepthNum ++;
            }
        }
    }


    for(int y=0; y<height; y++)
    {
        for(int x=0; x<width; x++)
        {
            float idepthValue = idepth[x+y*width];

            if(!std::isnan(idepthValue) && idepthValue > 0)
            {
                currentDepthMap[x+y*width] = DepthMapPixelHypothesis(
                                                 idepthValue,
                                                 idepthValue,
                                                 VAR_GT_INIT_INITIAL,
                                                 VAR_GT_INIT_INITIAL,
                                                 20);
            }
            else
            {
                currentDepthMap[x+y*width].isValid = false;
                currentDepthMap[x+y*width].blacklisted = 0;
            }
        }
    }


    activeKeyFrame->setDepth(currentDepthMap);
}

void DepthMap::resetCounters()
{
    runningStats.num_stereo_comparisons=0;
    runningStats.num_pixelInterpolations=0;
    runningStats.num_stereo_calls = 0;

    runningStats.num_stereo_rescale_oob = 0;
    runningStats.num_stereo_inf_oob = 0;
    runningStats.num_stereo_near_oob = 0;
    runningStats.num_stereo_invalid_unclear_winner = 0;
    runningStats.num_stereo_invalid_atEnd = 0;
    runningStats.num_stereo_invalid_inexistantCrossing = 0;
    runningStats.num_stereo_invalid_twoCrossing = 0;
    runningStats.num_stereo_invalid_noCrossing = 0;
    runningStats.num_stereo_invalid_bigErr = 0;
    runningStats.num_stereo_interpPre = 0;
    runningStats.num_stereo_interpPost = 0;
    runningStats.num_stereo_interpNone = 0;
    runningStats.num_stereo_negative = 0;
    runningStats.num_stereo_successfull = 0;

    runningStats.num_observe_created=0;
    runningStats.num_observe_create_attempted=0;
    runningStats.num_observe_updated=0;
    runningStats.num_observe_update_attempted=0;
    runningStats.num_observe_skipped_small_epl=0;
    runningStats.num_observe_skipped_small_epl_grad=0;
    runningStats.num_observe_skipped_small_epl_angle=0;
    runningStats.num_observe_transit_finalizing=0;
    runningStats.num_observe_transit_idle_oob=0;
    runningStats.num_observe_transit_idle_scale_angle=0;
    runningStats.num_observe_trans_idle_exhausted=0;
    runningStats.num_observe_inconsistent_finalizing=0;
    runningStats.num_observe_inconsistent=0;
    runningStats.num_observe_notfound_finalizing2=0;
    runningStats.num_observe_notfound_finalizing=0;
    runningStats.num_observe_notfound=0;
    runningStats.num_observe_skip_fail=0;
    runningStats.num_observe_skip_oob=0;
    runningStats.num_observe_good=0;
    runningStats.num_observe_good_finalizing=0;
    runningStats.num_observe_state_finalizing=0;
    runningStats.num_observe_state_initializing=0;
    runningStats.num_observe_skip_alreadyGood=0;
    runningStats.num_observe_addSkip=0;


    runningStats.num_observe_blacklisted=0;
}



void DepthMap::updateKeyframe(std::deque< std::shared_ptr<Frame> >
                              referenceFrames)
{
    assert(isValid());

    timepoint_t tv_start_all, tv_end_all;
    // gettimeofday(&tv_start_all, NULL);
    tv_start_all  = std::chrono::high_resolution_clock::now();

    oldest_referenceFrame = referenceFrames.front().get();
    newest_referenceFrame = referenceFrames.back().get();
    referenceFrameByID.clear();
    referenceFrameByID_offset = oldest_referenceFrame->id();

    for(std::shared_ptr<Frame> frame : referenceFrames)
    {
        assert(frame->hasTrackingParent());

        if(frame->getTrackingParent() != activeKeyFrame)
        {
            printf("WARNING: updating frame %d with %d, which was tracked on a different frame (%d).\nWhile this should work, it is not recommended.",
                   activeKeyFrame->id(), frame->id(),
                   frame->getTrackingParent()->id());
        }

        Sim3 refToKf;
        if(frame->pose->trackingParent->frameID == activeKeyFrame->id())
            refToKf = frame->pose->thisToParent_raw;
        else
            refToKf = activeKeyFrame->getScaledCamToWorld().inverse() *
                      frame->getScaledCamToWorld();

        frame->prepareForStereoWith(activeKeyFrame, refToKf, K, 0);

        while((int)referenceFrameByID.size() + referenceFrameByID_offset <=
                frame->id())
            referenceFrameByID.push_back(frame.get());
    }

    resetCounters();


    if(plotStereoImages)
    {
        cv::Mat keyFrameImage(activeKeyFrame->height(), activeKeyFrame->width(),
                              CV_32F, const_cast<float*>(activeKeyFrameImageData));
        keyFrameImage.convertTo(debugImageHypothesisHandling, CV_8UC1);
        cv::cvtColor(debugImageHypothesisHandling, debugImageHypothesisHandling,
                     CV_GRAY2RGB);

        cv::Mat oldest_refImage(oldest_referenceFrame->height(),
                                oldest_referenceFrame->width(), CV_32F,
                                const_cast<float*>(oldest_referenceFrame->image(0)));
        cv::Mat newest_refImage(newest_referenceFrame->height(),
                                newest_referenceFrame->width(), CV_32F,
                                const_cast<float*>(newest_referenceFrame->image(0)));
        cv::Mat rfimg = 0.5f*oldest_refImage + 0.5f*newest_refImage;
        rfimg.convertTo(debugImageStereoLines, CV_8UC1);
        cv::cvtColor(debugImageStereoLines, debugImageStereoLines, CV_GRAY2RGB);
    }

    timepoint_t tv_start, tv_end;


    // gettimeofday(&tv_start, NULL);
    tv_start = std::chrono::high_resolution_clock::now();
    observeDepth();
    // gettimeofday(&tv_end, NULL);
    tv_end = std::chrono::high_resolution_clock::now();
    msObserve = 0.9*msObserve +
                0.1*std::chrono::duration_cast<std::chrono::milliseconds>
                (tv_end - tv_start).count();
    nObserve++;

    //if(rand()%10==0)
    {
        // gettimeofday(&tv_start, NULL);
        tv_start = std::chrono::high_resolution_clock::now();
        regularizeDepthMapFillHoles();
        // gettimeofday(&tv_end, NULL);
        tv_end = std::chrono::high_resolution_clock::now();
        msFillHoles = 0.9*msFillHoles +
                      0.1*std::chrono::duration_cast<std::chrono::milliseconds>
                      (tv_end - tv_start).count();
        nFillHoles++;
    }


    // gettimeofday(&tv_start, NULL);
    tv_start = std::chrono::high_resolution_clock::now();
    regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
    // gettimeofday(&tv_end, NULL);
    tv_end = std::chrono::high_resolution_clock::now();
    msRegularize = 0.9*msRegularize +
                   0.1*std::chrono::duration_cast<std::chrono::milliseconds>
                   (tv_end - tv_start).count();
    nRegularize++;


    // Update depth in keyframe
    if(!activeKeyFrame->depthHasBeenUpdatedFlag)
    {
        // gettimeofday(&tv_start, NULL);
        tv_start = std::chrono::high_resolution_clock::now();
        activeKeyFrame->setDepth(currentDepthMap);
        // gettimeofday(&tv_end, NULL);
        tv_end = std::chrono::high_resolution_clock::now();
        msSetDepth = 0.9*msSetDepth +
                     0.1*std::chrono::duration_cast<std::chrono::milliseconds>
                     (tv_end - tv_start).count();
        nSetDepth++;
    }


    // gettimeofday(&tv_end_all, NULL);
    tv_end_all = std::chrono::high_resolution_clock::now();
    msUpdate = 0.9*msUpdate +
               0.1*std::chrono::duration_cast<std::chrono::milliseconds>
               (tv_end_all - tv_start_all).count();
    nUpdate++;


    activeKeyFrame->numMappedOnThis++;
    activeKeyFrame->numMappedOnThisTotal++;


    if(plotStereoImages)
    {
        Util::displayImage( "Stereo Key Frame", debugImageHypothesisHandling, false );
        Util::displayImage( "Stereo Reference Frame", debugImageStereoLines, false );
    }



    if(enablePrintDebugInfo && printLineStereoStatistics)
    {
        printf("ST: calls %6d, comp %6d, int %7d; good %6d (%.0f%%), neg %6d (%.0f%%); interp %6d / %6d / %6d\n",
               runningStats.num_stereo_calls,
               runningStats.num_stereo_comparisons,
               runningStats.num_pixelInterpolations,
               runningStats.num_stereo_successfull,
               100*runningStats.num_stereo_successfull / (float)
               runningStats.num_stereo_calls,
               runningStats.num_stereo_negative,
               100*runningStats.num_stereo_negative / (float)
               runningStats.num_stereo_successfull,
               runningStats.num_stereo_interpPre,
               runningStats.num_stereo_interpNone,
               runningStats.num_stereo_interpPost);
    }
    if(enablePrintDebugInfo && printLineStereoFails)
    {
        printf("ST-ERR: oob %d (scale %d, inf %d, near %d); err %d (%d uncl; %d end; zro: %d btw, %d no, %d two; %d big)\n",
               runningStats.num_stereo_rescale_oob+
               runningStats.num_stereo_inf_oob+
               runningStats.num_stereo_near_oob,
               runningStats.num_stereo_rescale_oob,
               runningStats.num_stereo_inf_oob,
               runningStats.num_stereo_near_oob,
               runningStats.num_stereo_invalid_unclear_winner+
               runningStats.num_stereo_invalid_atEnd+
               runningStats.num_stereo_invalid_inexistantCrossing+
               runningStats.num_stereo_invalid_noCrossing+
               runningStats.num_stereo_invalid_twoCrossing+
               runningStats.num_stereo_invalid_bigErr,
               runningStats.num_stereo_invalid_unclear_winner,
               runningStats.num_stereo_invalid_atEnd,
               runningStats.num_stereo_invalid_inexistantCrossing,
               runningStats.num_stereo_invalid_noCrossing,
               runningStats.num_stereo_invalid_twoCrossing,
               runningStats.num_stereo_invalid_bigErr);
    }
}

void DepthMap::invalidate()
{
    if(activeKeyFrame==0) return;
    activeKeyFrame=0;
    activeKeyFramelock.unlock();
}

void DepthMap::createKeyFrame(Frame* new_keyframe)
{
    assert(isValid());
    assert(new_keyframe != nullptr);
    assert(new_keyframe->hasTrackingParent());

    //boost::shared_lock<boost::shared_mutex> lock = activeKeyFrame->getActiveLock();
    boost::shared_lock<boost::shared_mutex> lock2 = new_keyframe->getActiveLock();

    timepoint_t tv_start_all, tv_end_all;
    // gettimeofday(&tv_start_all, NULL);
    tv_start_all = std::chrono::high_resolution_clock::now();


    resetCounters();

    if(plotStereoImages)
    {
        cv::Mat keyFrameImage(new_keyframe->height(), new_keyframe->width(), CV_32F,
                              const_cast<float*>(new_keyframe->image(0)));
        keyFrameImage.convertTo(debugImageHypothesisPropagation, CV_8UC1);
        cv::cvtColor(debugImageHypothesisPropagation, debugImageHypothesisPropagation,
                     CV_GRAY2RGB);
    }



    SE3 oldToNew_SE3 = se3FromSim3(new_keyframe->pose->thisToParent_raw).inverse();

    timepoint_t tv_start, tv_end;
    // gettimeofday(&tv_start, NULL);
    tv_start = std::chrono::high_resolution_clock::now();
    propagateDepth(new_keyframe);
    // gettimeofday(&tv_end, NULL);
    tv_end = std::chrono::high_resolution_clock::now();
    msPropagate = 0.9*msPropagate +
                  0.1*std::chrono::duration_cast<std::chrono::milliseconds>
                  (tv_end - tv_start).count();
    nPropagate++;

    activeKeyFrame = new_keyframe;
    activeKeyFramelock = activeKeyFrame->getActiveLock();
    activeKeyFrameImageData = new_keyframe->image(0);
    activeKeyFrameIsReactivated = false;



    // gettimeofday(&tv_start, NULL);
    tv_start = std::chrono::high_resolution_clock::now();
    regularizeDepthMap(true, VAL_SUM_MIN_FOR_KEEP);
    // gettimeofday(&tv_end, NULL);
    tv_end = std::chrono::high_resolution_clock::now();
    msRegularize = 0.9*msRegularize +
                   0.1*std::chrono::duration_cast<std::chrono::milliseconds>
                   (tv_end - tv_start).count();
    nRegularize++;


    // gettimeofday(&tv_start, NULL);
    tv_start = std::chrono::high_resolution_clock::now();
    regularizeDepthMapFillHoles();
    // gettimeofday(&tv_end, NULL);
    tv_end = std::chrono::high_resolution_clock::now();
    msFillHoles = 0.9*msFillHoles +
                  0.1*std::chrono::duration_cast<std::chrono::milliseconds>
                  (tv_end - tv_start).count();
    nFillHoles++;


    // gettimeofday(&tv_start, NULL);
    tv_start = std::chrono::high_resolution_clock::now();
    regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
    // gettimeofday(&tv_end, NULL);
    tv_end = std::chrono::high_resolution_clock::now();
    msRegularize = 0.9*msRegularize +
                   0.1*std::chrono::duration_cast<std::chrono::milliseconds>
                   (tv_end - tv_start).count();
    nRegularize++;




    // make mean inverse depth be one.
    float sumIdepth=0, numIdepth=0;
    for(DepthMapPixelHypothesis* source = currentDepthMap;
            source < currentDepthMap+width*height; source++)
    {
        if(!source->isValid)
            continue;
        sumIdepth += source->idepth_smoothed;
        numIdepth++;
    }
    float rescaleFactor = numIdepth / sumIdepth;
    float rescaleFactor2 = rescaleFactor*rescaleFactor;
    for(DepthMapPixelHypothesis* source = currentDepthMap;
            source < currentDepthMap+width*height; source++)
    {
        if(!source->isValid)
            continue;
        source->idepth *= rescaleFactor;
        source->idepth_smoothed *= rescaleFactor;
        source->idepth_var *= rescaleFactor2;
        source->idepth_var_smoothed *= rescaleFactor2;
    }
    activeKeyFrame->pose->thisToParent_raw = sim3FromSE3(oldToNew_SE3.inverse(),
            rescaleFactor);
    activeKeyFrame->pose->invalidateCache();

    // Update depth in keyframe

    // gettimeofday(&tv_start, NULL);
    tv_start = std::chrono::high_resolution_clock::now();
    activeKeyFrame->setDepth(currentDepthMap);
    // gettimeofday(&tv_end, NULL);
    tv_end = std::chrono::high_resolution_clock::now();
    msSetDepth = 0.9*msSetDepth +
                 0.1*std::chrono::duration_cast<std::chrono::milliseconds>
                 (tv_end - tv_start).count();
    nSetDepth++;

    //gettimeofday(&tv_end_all, NULL);
    tv_end_all = std::chrono::high_resolution_clock::now();
    msCreate = 0.9*msCreate +
               0.1*std::chrono::duration_cast<std::chrono::milliseconds>
               (tv_end_all - tv_start_all).count();
    nCreate++;



    if(plotStereoImages)
    {
        //Util::displayImage( "KeyFramePropagation", debugImageHypothesisPropagation );
    }

}

void DepthMap::addTimingSample()
{
    timepoint_t now;
    // gettimeofday(&now, NULL);
    now = std::chrono::high_resolution_clock::now();
    // float sPassed = ((now.tv_sec-lastHzUpdate.tv_sec) + (now.tv_usec-lastHzUpdate.tv_usec)/1000000.0f);
    float sPassed = std::chrono::duration_cast<std::chrono::seconds>
                    (now - lastHzUpdate).count();
    if(sPassed > 1.0f)
    {
        nAvgUpdate = 0.8*nAvgUpdate + 0.2*(nUpdate / sPassed);
        nUpdate = 0;
        nAvgCreate = 0.8*nAvgCreate + 0.2*(nCreate / sPassed);
        nCreate = 0;
        nAvgFinalize = 0.8*nAvgFinalize + 0.2*(nFinalize / sPassed);
        nFinalize = 0;
        nAvgObserve = 0.8*nAvgObserve + 0.2*(nObserve / sPassed);
        nObserve = 0;
        nAvgRegularize = 0.8*nAvgRegularize + 0.2*(nRegularize / sPassed);
        nRegularize = 0;
        nAvgPropagate = 0.8*nAvgPropagate + 0.2*(nPropagate / sPassed);
        nPropagate = 0;
        nAvgFillHoles = 0.8*nAvgFillHoles + 0.2*(nFillHoles / sPassed);
        nFillHoles = 0;
        nAvgSetDepth = 0.8*nAvgSetDepth + 0.2*(nSetDepth / sPassed);
        nSetDepth = 0;
        lastHzUpdate = now;

        if(enablePrintDebugInfo && printMappingTiming)
        {
            printf("Upd %3.1fms (%.1fHz); Create %3.1fms (%.1fHz); Final %3.1fms (%.1fHz) // Obs %3.1fms (%.1fHz); Reg %3.1fms (%.1fHz); Prop %3.1fms (%.1fHz); Fill %3.1fms (%.1fHz); Set %3.1fms (%.1fHz)\n",
                   msUpdate, nAvgUpdate,
                   msCreate, nAvgCreate,
                   msFinalize, nAvgFinalize,
                   msObserve, nAvgObserve,
                   msRegularize, nAvgRegularize,
                   msPropagate, nAvgPropagate,
                   msFillHoles, nAvgFillHoles,
                   msSetDepth, nAvgSetDepth);
        }
    }


}

void DepthMap::finalizeKeyFrame()
{
    assert(isValid());


    timepoint_t tv_start_all, tv_end_all;
    //gettimeofday(&tv_start_all, NULL);
    tv_start_all = std::chrono::high_resolution_clock::now();
    timepoint_t tv_start, tv_end;

    // gettimeofday(&tv_start, NULL);
    tv_start = std::chrono::high_resolution_clock::now();
    regularizeDepthMapFillHoles();
    // gettimeofday(&tv_end, NULL);
    tv_end = std::chrono::high_resolution_clock::now();
    msFillHoles = 0.9*msFillHoles +
                  0.1*std::chrono::duration_cast<std::chrono::milliseconds>
                  (tv_end - tv_start).count();
    nFillHoles++;

    // gettimeofday(&tv_start, NULL);
    tv_start = std::chrono::high_resolution_clock::now();
    regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
    // gettimeofday(&tv_end, NULL);
    tv_end = std::chrono::high_resolution_clock::now();
    msRegularize = 0.9*msRegularize +
                   0.1*std::chrono::duration_cast<std::chrono::milliseconds>
                   (tv_end - tv_start).count();
    nRegularize++;

    // gettimeofday(&tv_start, NULL);
    tv_start = std::chrono::high_resolution_clock::now();
    activeKeyFrame->setDepth(currentDepthMap);
    activeKeyFrame->calculateMeanInformation();
    activeKeyFrame->takeReActivationData(currentDepthMap);
    // gettimeofday(&tv_end, NULL);
    tv_end = std::chrono::high_resolution_clock::now();
    msSetDepth = 0.9*msSetDepth +
                 0.1*std::chrono::duration_cast<std::chrono::milliseconds>
                 (tv_end - tv_start).count();
    nSetDepth++;

    //gettimeofday(&tv_end_all, NULL);
    tv_end_all = std::chrono::high_resolution_clock::now();
    msFinalize = 0.9*msFinalize +
                 0.1*std::chrono::duration_cast<std::chrono::milliseconds>
                 (tv_end_all - tv_start_all).count();
    nFinalize++;
}




int DepthMap::debugPlotDepthMap()
{
    if(activeKeyFrame == 0) return 1;

    cv::Mat keyFrameImage(activeKeyFrame->height(), activeKeyFrame->width(),
                          CV_32F, const_cast<float*>(activeKeyFrameImageData));
    keyFrameImage.convertTo(debugImageDepth, CV_8UC1);
    cv::cvtColor(debugImageDepth, debugImageDepth, CV_GRAY2RGB);

    // debug plot & publish sparse version?
    int refID = referenceFrameByID_offset;


    for(int y=0; y<height; y++)
        for(int x=0; x<width; x++)
        {
            int idx = x + y*width;

            if(currentDepthMap[idx].blacklisted < MIN_BLACKLIST && debugDisplay == 2)
                debugImageDepth.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,255);

            if(!currentDepthMap[idx].isValid) continue;

            cv::Vec3b color = currentDepthMap[idx].getVisualizationColor(refID);
            debugImageDepth.at<cv::Vec3b>(y,x) = color;
        }


    return 1;
}


bool isInExclusiveRange(float x, float low, float high) {
    return low < x && x < high;
}


bool DepthMap::isIn2Sigma(const Eigen::Vector2f &p, const Eigen::Vector2f &epn, const float rescaleFactor) {
    const Eigen::Vector2f L = p - 2*epn*rescaleFactor;
    const Eigen::Vector2f H = p + 2*epn*rescaleFactor;
    // width - 2 and height - 2 comes from the one-sided gradient calculation at the bottom
    return 0 < L(0) && L(0) < width - 2 && 0 < L(1) && L(1) < height - 2 &&
           0 < H(0) && H(0) < width - 2 && 0 < H(1) && H(1) < height - 2;
}


bool DepthMap::isInExclusiveImageRange(Eigen::Vector2f x, float margin) {
    return isInExclusiveRange(x(0), margin, width-margin) &&
           isInExclusiveRange(x(1), margin, height-margin);
}

inline Eigen::VectorXf calcGrad(const Eigen::VectorXf &intensities) {
    const int n = intensities.size() - 1;
    return intensities.segment(1, n) - intensities.segment(0, n);
}


// find pixel in image (do stereo along epipolar line).
// mat: NEW image
// KinvP: point in OLD image (Kinv * (u_old, v_old, 1)), projected
// trafo: x_old = trafo * x_new; (from new to old image)
// realVal: descriptor in OLD image.
// returns: result_idepth : point depth in new camera's coordinate system
// returns: result_u/p(1) : point's coordinates in new camera's coordinate system
// returns: idepth_var: (approximated) measurement variance of inverse depth of result_point_NEW
// returns error if sucessful; -1 if out of bounds, -2 if not found.
float DepthMap::doLineStereo(
    const Eigen::Vector2f &p, const Eigen::Vector2f epn,
    const float min_idepth, const float prior_idepth, float max_idepth,
    const Frame* const referenceFrame, const float* referenceFrameImage,
    float &result_idepth, float &result_var, float &result_eplLength,
    RunningStats* stats)
{
    if(enablePrintDebugInfo) stats->num_stereo_calls++;

    // calculate epipolar line start and end point in old image
    Eigen::Vector3f KinvP = Eigen::Vector3f(fxi*p(0)+cxi,fyi*p(1)+cyi,1);
    Eigen::Vector3f pInf = referenceFrame->K_otherToThis_R * KinvP;
    Eigen::Vector3f pReal = pInf / prior_idepth + referenceFrame->K_otherToThis_t;

    float rescaleFactor = pReal(2) * prior_idepth;

    if(!isIn2Sigma(p, epn, rescaleFactor)) {
        return -1;
    }

    if(!isInExclusiveRange(rescaleFactor, 0.7f, 1.4f))
    {
        if(enablePrintDebugInfo) stats->num_stereo_rescale_oob++;
        return -1;
    }

    // calculate values to search for
    Eigen::VectorXf realVal(5);
    realVal(0) = getInterpolatedElement(activeKeyFrameImageData, p - 2*epn*rescaleFactor, width);
    realVal(1) = getInterpolatedElement(activeKeyFrameImageData, p - epn*rescaleFactor, width);
    realVal(2) = getInterpolatedElement(activeKeyFrameImageData, p, width);
    realVal(3) = getInterpolatedElement(activeKeyFrameImageData, p + epn*rescaleFactor, width);
    realVal(4) = getInterpolatedElement(activeKeyFrameImageData, p + 2*epn*rescaleFactor, width);

    Eigen::Vector3f PClose = pInf + referenceFrame->K_otherToThis_t*max_idepth;
    Eigen::Vector3f PFar = pInf + referenceFrame->K_otherToThis_t*min_idepth;

    // if the assumed close-point lies behind the
    // image, have to change that.
    if(PClose[2] < 0.001)
    {
        max_idepth = (0.001-pInf[2]) / referenceFrame->K_otherToThis_t[2];
        PClose = pInf + referenceFrame->K_otherToThis_t*max_idepth;
    }

    // if the assumed far-point lies behind the image or closter than the near-point,
    // we moved past the Point it and should stop.
    if(PFar[2] < 0.001 || max_idepth < min_idepth)
    {
        if(enablePrintDebugInfo) stats->num_stereo_inf_oob++;
        return -1;
    }

    // pos in new image of point (xy), assuming max_idepth
    Eigen::Vector2f pClose = projection(PClose);
    Eigen::Vector2f pFar = projection(PFar); // pos in new image of point (xy), assuming min_idepth

    // check for nan due to eg division by zero.
    if(std::isnan((float)(pFar[0]+pClose[0])))
        return -4;

    // calculate increments in which we will step through the epipolar line.
    // they are sampleDist (or half sample dist) long
    Eigen::Vector2f inc = pClose - pFar;

    float eplLength = inc.norm();
    if(!eplLength > 0 || std::isinf(eplLength)) return -4;

    if(eplLength > MAX_EPL_LENGTH_CROP)
    {
        pClose = pFar + inc*MAX_EPL_LENGTH_CROP/eplLength;
    }

    inc = inc * GRADIENT_SAMPLE_DIST/eplLength;

    // extend one sample_dist to left & right.
    pFar -= inc;
    pClose += inc;

    // make epl long enough (pad a little bit).
    if(eplLength < MIN_EPL_LENGTH_CROP)
    {
        float pad = (MIN_EPL_LENGTH_CROP - (eplLength)) / 2;
        pFar -= inc*pad;
        pClose += inc*pad;
    }

    // if inf point is outside of image: skip pixel.
    if(!isInExclusiveImageRange(pFar, SAMPLE_POINT_TO_BORDER))
    {
        if(enablePrintDebugInfo) stats->num_stereo_inf_oob++;
        return -1;
    }

    // if near point is outside: move inside, and test length again.
    if(!isInExclusiveImageRange(pClose, SAMPLE_POINT_TO_BORDER))
    {
        if(pClose(0) <= SAMPLE_POINT_TO_BORDER)
        {
            float toAdd = (SAMPLE_POINT_TO_BORDER - pClose(0)) / inc(0);
            pClose += toAdd * inc;
        }
        else if(pClose(0) >= width-SAMPLE_POINT_TO_BORDER)
        {
            float toAdd = (width-SAMPLE_POINT_TO_BORDER - pClose(0)) / inc(0);
            pClose += toAdd * inc;
        }

        if(pClose(1) <= SAMPLE_POINT_TO_BORDER)
        {
            float toAdd = (SAMPLE_POINT_TO_BORDER - pClose(1)) / inc(1);
            pClose += toAdd * inc;
        }
        else if(pClose(1) >= height-SAMPLE_POINT_TO_BORDER)
        {
            float toAdd = (height-SAMPLE_POINT_TO_BORDER - pClose(1)) / inc(1);
            pClose += toAdd * inc;
        }

        // get new epl length
        float newEplLength = (pClose - pFar).norm();

        // test again
        if(!isInExclusiveImageRange(pClose, SAMPLE_POINT_TO_BORDER) || newEplLength < 8)
        {
            if(enablePrintDebugInfo) stats->num_stereo_near_oob++;
            return -1;
        }
    }


    // from here on:
    // - pInf: search start-point
    // - p0: search end-point
    // - inc(0), inc(1): search steps in pixel
    // - eplLength, min_idepth, max_idepth: determines search-resolution, i.e. the result's variance.


    Eigen::Vector2f cp = pFar;

    Eigen::VectorXf val_cp(5);
    val_cp(0) = getInterpolatedElement(referenceFrameImage, cp-2*inc, width);
    val_cp(1) = getInterpolatedElement(referenceFrameImage, cp-inc, width);
    val_cp(2) = getInterpolatedElement(referenceFrameImage, cp, width);
    val_cp(3) = getInterpolatedElement(referenceFrameImage, cp+inc, width);

    /*
     * Subsequent exact minimum is found the following way:
     * - assuming lin. interpolation, the gradient of Error at p1 (towards p2) is given by
     *   dE1 = -2sum(e1*e1 - e1*e2)
     *   where e1 and e2 are summed over, and are the residuals (not squared).
     *
     * - the gradient at p2 (coming from p1) is given by
     * 	 dE2 = +2sum(e2*e2 - e1*e2)
     *
     * - linear interpolation => gradient changes linearely; zero-crossing is hence given by
     *   p1 + d*(p2-p1) with d = -dE1 / (-dE1 + dE2).
     *
     *
     *
     * => I for later exact min calculation, I need sum(e_i*e_i),sum(e_{i-1}*e_{i-1}),sum(e_{i+1}*e_{i+1})
     *    and sum(e_i * e_{i-1}) and sum(e_i * e_{i+1}),
     *    where i is the respective winning index.
     */


    // walk in equally sized steps, starting at depth=infinity.
    int index = 0;
    Eigen::Vector2f best_match;
    float best_match_err = 1e50;
    float second_best_match_err = 1e50;

    // best pre and post errors.
    float best_match_err_pre=NAN, best_match_err_post=NAN;
    float best_match_diff_error_pre=NAN, best_match_diff_error_post=NAN;

    float eeLast = -1; // final error of last comp.

    int arg_best=-1, arg_second_best =-1;
    while(((inc(0) < 0) == (cp(0) > pClose[0]) && (inc(1) < 0) == (cp(1) > pClose[1])) ||
          index == 0)
    {
        // interpolate one new point
        val_cp(4) = getInterpolatedElement(referenceFrameImage, cp+2*inc, width);

        // hacky but fast way to get error and differential error: switch buffer variables for last loop.
        float ee = 0;
        Eigen::VectorXf eA(5);
        Eigen::VectorXf eB(5);
        if(index%2==0)
        {
            eA = val_cp - realVal;
            ee += eA.dot(eA);
        }
        else
        {
            eB = val_cp - realVal;
            ee += eB.dot(eB);
        }


        // do I have a new winner??
        // if so: set.
        if(ee < best_match_err)
        {
            // put to second-best
            second_best_match_err=best_match_err;
            arg_second_best = arg_best;

            // set best.
            best_match_err = ee;
            arg_best = index;

            best_match_err_pre = eeLast;
            best_match_err_post = -1;
            best_match_diff_error_pre = eA.dot(eB);
            best_match_diff_error_post = -1;

            best_match = cp;
        }
        // otherwise: the last might be the current winner, in which case i have to save these values.
        else
        {
            if(index == arg_best + 1)
            {
                best_match_err_post = ee;
                best_match_diff_error_post = eA.dot(eB);
            }

            // collect second-best:
            // just take the best of all that are NOT equal to current best.
            if(ee < second_best_match_err)
            {
                second_best_match_err=ee;
                arg_second_best = index;
            }
        }


        // shift everything one further.
        eeLast = ee;
        val_cp.segment(0, 4) = val_cp.segment(1, 4);

        if(enablePrintDebugInfo) stats->num_stereo_comparisons++;

        cp += inc;

        index++;
    }

    // if error too big, will return -3, otherwise -2.
    if(best_match_err > 4*(float)MAX_ERROR_STEREO)
    {
        if(enablePrintDebugInfo) stats->num_stereo_invalid_bigErr++;
        return -3;
    }


    // check if clear enough winner
    if(abs(arg_best - arg_second_best) > 1 &&
           MIN_DISTANCE_ERROR_STEREO * best_match_err > second_best_match_err)
    {
        if(enablePrintDebugInfo) stats->num_stereo_invalid_unclear_winner++;
        return -2;
    }

    bool didSubpixel = false;
    if(useSubpixelStereo)
    {
        // ================== compute exact match =========================
        // compute gradients (they are actually only half the real gradient)
        float gradPre_pre = -(best_match_err_pre - best_match_diff_error_pre);
        float gradPre_this = +(best_match_err - best_match_diff_error_pre);
        float gradPost_this = -(best_match_err - best_match_diff_error_post);
        float gradPost_post = +(best_match_err_post - best_match_diff_error_post);

        // final decisions here.
        bool interpPost = false;
        bool interpPre = false;

        // if one is oob: return false.
        if(enablePrintDebugInfo && (best_match_err_pre < 0 || best_match_err_post < 0))
        {
            stats->num_stereo_invalid_atEnd++;
        }


        // - if zero-crossing occurs exactly in between (gradient Inconsistent),
        else if((gradPost_this < 0) ^ (gradPre_this < 0))
        {
            // return exact pos, if both central gradients are small compared to their counterpart.
            if(enablePrintDebugInfo
                    && (gradPost_this*gradPost_this > 0.1f*0.1f*gradPost_post*gradPost_post ||
                        gradPre_this*gradPre_this > 0.1f*0.1f*gradPre_pre*gradPre_pre))
                stats->num_stereo_invalid_inexistantCrossing++;
        }

        // if pre has zero-crossing
        else if((gradPre_pre < 0) ^ (gradPre_this < 0))
        {
            // if post has zero-crossing
            if((gradPost_post < 0) ^ (gradPost_this < 0))
            {
                if(enablePrintDebugInfo) stats->num_stereo_invalid_twoCrossing++;
            }
            else
                interpPre = true;
        }

        // if post has zero-crossing
        else if((gradPost_post < 0) ^ (gradPost_this < 0))
        {
            interpPost = true;
        }

        // if none has zero-crossing
        else
        {
            if(enablePrintDebugInfo) stats->num_stereo_invalid_noCrossing++;
        }


        // DO interpolation!
        // minimum occurs at zero-crossing of gradient, which is a straight line => easy to compute.
        // the error at that point is also computed by just integrating.
        if(interpPre)
        {
            float d = gradPre_this / (gradPre_this - gradPre_pre);
            best_match -= d*inc;
            best_match_err = best_match_err - 2*d*gradPre_this - (gradPre_pre -
                             gradPre_this)*d*d;
            if(enablePrintDebugInfo) stats->num_stereo_interpPre++;
            didSubpixel = true;

        }
        else if(interpPost)
        {
            float d = gradPost_this / (gradPost_this - gradPost_post);
            best_match += d*inc;
            best_match_err = best_match_err + 2*d*gradPost_this + (gradPost_post -
                             gradPost_this)*d*d;
            if(enablePrintDebugInfo) stats->num_stereo_interpPost++;
            didSubpixel = true;
        }
        else
        {
            if(enablePrintDebugInfo) stats->num_stereo_interpNone++;
        }
    }


    // sampleDist is the distance in pixel at which the realVal's were sampled
    float sampleDist = GRADIENT_SAMPLE_DIST*rescaleFactor;

    Eigen::VectorXf grad = calcGrad(realVal);
    float gradAlongLine = grad.dot(grad);
    gradAlongLine /= sampleDist*sampleDist;

    // check if interpolated error is OK. use evil hack to allow more error if there is a lot of gradient.
    if(best_match_err > (float)MAX_ERROR_STEREO + sqrtf(gradAlongLine)*20)
    {
        if(enablePrintDebugInfo) stats->num_stereo_invalid_bigErr++;
        return -3;
    }


    // ================= calc depth (in KF) ====================
    // * KinvP = Kinv * (x,y,1); where x,y are pixel coordinates of point we search for, in the KF.
    // * best_match_x = x-coordinate of found correspondence in the reference frame.

    float idnew_best_match;	// depth in the new image
    float alpha; // d(idnew_best_match) / d(disparity in pixel) == conputed inverse depth derived by the pixel-disparity.
    if(inc(0)*inc(0)>inc(1)*inc(1))
    {
        float oldX = fxi*best_match(0)+cxi;
        float nominator = (oldX*referenceFrame->otherToThis_t[2] -
                           referenceFrame->otherToThis_t[0]);
        float dot0 = KinvP.dot(referenceFrame->otherToThis_R_row0);
        float dot2 = KinvP.dot(referenceFrame->otherToThis_R_row2);

        idnew_best_match = (dot0 - oldX*dot2) / nominator;
        alpha = inc(0)*fxi*(dot0*referenceFrame->otherToThis_t[2] -
                          dot2*referenceFrame->otherToThis_t[0]) / (nominator*nominator);

    }
    else
    {
        float oldY = fyi*best_match(1)+cyi;

        float nominator = (oldY*referenceFrame->otherToThis_t[2] -
                           referenceFrame->otherToThis_t[1]);
        float dot1 = KinvP.dot(referenceFrame->otherToThis_R_row1);
        float dot2 = KinvP.dot(referenceFrame->otherToThis_R_row2);

        idnew_best_match = (dot1 - oldY*dot2) / nominator;
        alpha = inc(1)*fyi*(dot1*referenceFrame->otherToThis_t[2] -
                          dot2*referenceFrame->otherToThis_t[1]) / (nominator*nominator);

    }

    if(idnew_best_match < 0)
    {
        if(enablePrintDebugInfo) stats->num_stereo_negative++;
        if(!allowNegativeIdepths)
            return -2;
    }

    if(enablePrintDebugInfo) stats->num_stereo_successfull++;

    // ================= calc var (in NEW image) ====================

    // calculate error from photometric noise
    float photoDispError = 4 * cameraPixelNoise2 / (gradAlongLine + DIVISION_EPS);

    float trackingErrorFac = 0.25*(1+referenceFrame->initialTrackedResidual);

    // calculate error from geometric noise (wrong camera pose / calibration)
    Eigen::Vector2f gradsInterp = getInterpolatedElement42(activeKeyFrame->gradients(0), p, width);
    float geoDispError = gradsInterp.dot(epn) + DIVISION_EPS;
    geoDispError = trackingErrorFac*trackingErrorFac*gradsInterp.dot(gradsInterp) / (geoDispError*geoDispError);


    //geoDispError *= (0.5 + 0.5 *result_idepth) * (0.5 + 0.5 *result_idepth);

    // final error consists of a small constant part (discretization error),
    // geometric and photometric error.
    result_var = alpha*alpha*((didSubpixel ? 0.05f : 0.5f)*sampleDist*sampleDist +
                              geoDispError + photoDispError);	// square to make variance

    if(plotStereoImages)
    {
        if(rand()%5==0)
        {
            //if(rand()%500 == 0)
            //	printf("geo: %f, photo: %f, alpha: %f\n", sqrt(geoDispError), sqrt(photoDispError), alpha, sqrt(result_var));


            //int idDiff = (keyFrame->pyramidID - referenceFrame->id);
            //cv::Scalar color = cv::Scalar(0,0, 2*idDiff);// bw

            //cv::Scalar color = cv::Scalar(sqrt(result_var)*2000, 255-sqrt(result_var)*2000, 0);// bw

//			float eplLengthF = std::min((float)MIN_EPL_LENGTH_CROP,(float)eplLength);
//			eplLengthF = std::max((float)MAX_EPL_LENGTH_CROP,(float)eplLengthF);
//
//			float pixelDistFound = sqrtf((float)((pReal[0]/pReal[2] - best_match_x)*(pReal[0]/pReal[2] - best_match_x)
//					+ (pReal[1]/pReal[2] - best_match_y)*(pReal[1]/pReal[2] - best_match_y)));
//
            float fac = best_match_err / ((float)MAX_ERROR_STEREO + sqrtf(
                                              gradAlongLine)*20);

            cv::Scalar color = cv::Scalar(255*fac, 255-255*fac, 0);// bw


            /*
            if(rescaleFactor > 1)
            	color = cv::Scalar(500*(rescaleFactor-1),0,0);
            else
            	color = cv::Scalar(0,500*(1-rescaleFactor),500*(1-rescaleFactor));
            */

            cv::line(debugImageStereoLines,cv::Point2f(pClose[0], pClose[1]),
                     cv::Point2f(pFar[0], pFar[1]),color,1,8,0);
        }
    }

    result_idepth = idnew_best_match;

    result_eplLength = eplLength;

    return best_match_err;
}

}
