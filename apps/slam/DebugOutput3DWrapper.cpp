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

#include <cmath>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#define GL_GLEXT_PROTOTYPES

#include <GL/glut.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include "DebugOutput3DWrapper.h"
#include "lsd_slam/util/sophus_util.h"
#include "lsd_slam/util/settings.h"

//#include "lsd_slam_viewer/keyframeGraphMsg.h"
//#include "lsd_slam_viewer/keyframeMsg.h"

#include "sophus/sim3.hpp"
#include "sophus/se3.hpp"
#include "lsd_slam/io_wrapper/output_3d_wrapper.h"
#include "lsd_slam/model/frame.h"
#include "lsd_slam/global_mapping/key_frame_graph.h"
#include "lsd_slam/global_mapping/g2o_type_sim3_sophus.h"
#include "lsd_slam/projection/projection.h"


namespace lsd_slam
{


DebugOutput3DWrapper::DebugOutput3DWrapper(
    const int width_, const int height_, const int publishLvl_) :
    width(width_), height(height_), publishLvl(publishLvl_) {}

DebugOutput3DWrapper::~DebugOutput3DWrapper()
{
}


void init_rgbpoint(pcl::PointXYZRGB &point, const Eigen::Vector3f P, const cv::Vec3b color) {
    uint32_t rgb = ((uint32_t)color(0) << 16 | (uint32_t)color(1) << 8 | (uint32_t)color(2));

    point.x = P(0);
    point.y = P(1);
    point.z = P(2);

    point.rgb = *reinterpret_cast<float*>(&rgb);
}


inline bool isReliable(const float idepth, const float idepth_var, const float threshold = 0.03) {
    if(idepth <= 0) {
        return false;
    }

    const float idepth_std = sqrt(idepth_var);
    const float depth_squared = 1 / (idepth * idepth);  // = (1 / idepth) * (1 / idepth)

    return idepth_std * depth_squared <= threshold;
}

void DebugOutput3DWrapper::addPointsToPointCloud(
        pcl::PointCloud<pcl::PointXYZRGB> &pointcloud,
        const Projection &projection, const Sim3 &camToWorld,
        const int id, const float *idepth, const float *idepth_var) {

    const cv::Mat image = getImage(id);

    int n_added = 0;
    for(int v = 0; v < height; v++) {
        for(int u = 0; u < width; u++) {
            const int index = width * v + u;
            // TODO check idepthVar

            if(!isReliable(idepth[index], idepth_var[index], 0.05)) {
                continue;
            }

            Eigen::Matrix<float, 3,1> P;
            P = projection.inv_projection(u, v, 1/idepth[index]);
            P = (camToWorld * P.cast<double>()).cast<float>();

            pcl::PointXYZRGB point;
            init_rgbpoint(point, P, image.at<cv::Vec3b>(v, u));
            pointcloud.push_back(point);

            n_added += 1;
        }
    }

    std::cout << 100 * (float)n_added / (float)(height * width) << " % of points added" << std::endl;
}

void DebugOutput3DWrapper::savePointCloud(std::string filename) {
    std::cout << "Exporting pointcloud to " << filename << std::endl;

    pcl::io::savePLYFile(filename, this->pointcloud);
}


void DebugOutput3DWrapper::publishKeyframe(Frame* f)
{
	boost::shared_lock<boost::shared_mutex> lock = f->getActiveLock();

	int width = f->width(publishLvl);
	int height = f->height(publishLvl);

	Sim3 camToWorld = f->getScaledCamToWorld();

    Projection projection(f->fx(publishLvl), f->fy(publishLvl),
                          f->cx(publishLvl), f->cy(publishLvl));

    addPointsToPointCloud(this->pointcloud, projection, camToWorld,
                          f->id(), f->idepth(publishLvl), f->idepthVar(publishLvl));

	std::cout << "PublishKeyframe" << std::endl;
}

void DebugOutput3DWrapper::publishTrackedFrame(Frame* kf)
{
	KeyFrameMessage fMsg;


	fMsg.id = kf->id();
	fMsg.time = kf->timestamp();
	fMsg.isKeyframe = false;


	memcpy(fMsg.camToWorld.data(),kf->getScaledCamToWorld().cast<float>().data(),sizeof(float)*7);
	fMsg.fx = kf->fx(publishLvl);
	fMsg.fy = kf->fy(publishLvl);
	fMsg.cx = kf->cx(publishLvl);
	fMsg.cy = kf->cy(publishLvl);
	fMsg.width = kf->width(publishLvl);
	fMsg.height = kf->height(publishLvl);

	/*fMsg.pointcloud.clear();

	liveframe_publisher.publish(fMsg);*/


	SE3 camToWorld = se3FromSim3(kf->getScaledCamToWorld());

	/*geometry_msgs::PoseStamped pMsg;

	pMsg.pose.position.x = camToWorld.translation()[0];
	pMsg.pose.position.y = camToWorld.translation()[1];
	pMsg.pose.position.z = camToWorld.translation()[2];
	pMsg.pose.orientation.x = camToWorld.so3().unit_quaternion().x();
	pMsg.pose.orientation.y = camToWorld.so3().unit_quaternion().y();
	pMsg.pose.orientation.z = camToWorld.so3().unit_quaternion().z();
	pMsg.pose.orientation.w = camToWorld.so3().unit_quaternion().w();

	if (pMsg.pose.orientation.w < 0)
	{
		pMsg.pose.orientation.x *= -1;
		pMsg.pose.orientation.y *= -1;
		pMsg.pose.orientation.z *= -1;
		pMsg.pose.orientation.w *= -1;
	}

	pMsg.header.stamp = ros::Time(kf->timestamp());
	pMsg.header.frame_id = "world";
	pose_publisher.publish(pMsg);*/
}



void DebugOutput3DWrapper::publishKeyframeGraph(KeyFrameGraph* graph)
{
	/*lsd_slam_viewer::keyframeGraphMsg gMsg;

	graph->edgesListsMutex.lock();
	gMsg.numConstraints = graph->edgesAll.size();
	gMsg.constraintsData.resize(gMsg.numConstraints * sizeof(GraphConstraint));
	GraphConstraint* constraintData = (GraphConstraint*)gMsg.constraintsData.data();
	for(unsigned int i=0;i<graph->edgesAll.size();i++)
	{
		constraintData[i].from = graph->edgesAll[i]->firstFrame->id();
		constraintData[i].to = graph->edgesAll[i]->secondFrame->id();
		Sophus::Vector7d err = graph->edgesAll[i]->edge->error();
		constraintData[i].err = sqrt(err.dot(err));
	}
	graph->edgesListsMutex.unlock();

	graph->keyframesAllMutex.lock_shared();
	gMsg.numFrames = graph->keyframesAll.size();
	gMsg.frameData.resize(gMsg.numFrames * sizeof(GraphFramePose));
	GraphFramePose* framePoseData = (GraphFramePose*)gMsg.frameData.data();
	for(unsigned int i=0;i<graph->keyframesAll.size();i++)
	{
		framePoseData[i].id = graph->keyframesAll[i]->id();
		memcpy(framePoseData[i].camToWorld, graph->keyframesAll[i]->getScaledCamToWorld().cast<float>().data(),sizeof(float)*7);
	}
	graph->keyframesAllMutex.unlock_shared();

	graph_publisher.publish(gMsg);*/
}

void DebugOutput3DWrapper::publishTrajectory(std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> trajectory, std::string identifier)
{
	// unimplemented ... do i need it?
}

void DebugOutput3DWrapper::publishTrajectoryIncrement(const Eigen::Matrix<float, 3, 1>& pt, std::string identifier)
{
	// unimplemented ... do i need it?
}

void DebugOutput3DWrapper::publishDebugInfo(const Eigen::Matrix<float, 20, 1>& data)
{
	//std_msgs::Float32MultiArray msg;
	for(int i=0;i<20;i++)
		std::cout << (float)(data[i]) << std::endl;

	//debugInfo_publisher.publish(msg);
}

//void draw_target(cv::Mat& rgb_img, look3d::PanoramicTracker& tracker) {
//	const Eigen::Vector4d point_x(0.1, 0, 1, 1);
//	const Eigen::Vector4d point_y(0, 0.1, 1, 1);
//	const Eigen::Vector4d point_z(0, 0, 1.1, 1);
//	const Eigen::Vector4d point_target(0, 0, 1.0, 1);
//
//	Eigen::Matrix<double, 3, 4> proj = get_projection(tracker);
//
//	Eigen::Vector3d point_cam = proj * point_target;
//	Eigen::Vector3d pointx_cam = proj * point_x;
//	Eigen::Vector3d pointy_cam = proj * point_y;
//	Eigen::Vector3d pointz_cam = proj * point_z;
//
//	cv::line(rgb_img, cv::Point(point_cam[0], point_cam[1]), cv::Point(pointx_cam[0], pointx_cam[1]), cv::Scalar(255, 0, 0), 3);
//	cv::line(rgb_img, cv::Point(point_cam[0], point_cam[1]), cv::Point(pointy_cam[0], pointy_cam[1]), cv::Scalar(0, 255, 0), 3);
//	cv::line(rgb_img, cv::Point(point_cam[0], point_cam[1]), cv::Point(pointz_cam[0], pointz_cam[1]), cv::Scalar(0, 0, 255), 3);
//}

}
