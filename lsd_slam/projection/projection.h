#ifndef PROJECTION_PROJECTION_H
#define PROJECTION_PROJECTION_H

#include <Eigen/Core>


inline Eigen::Vector2f to2d(const Eigen::Vector3f &p) {
    return p.segment(0, 2) / p(2);
}
