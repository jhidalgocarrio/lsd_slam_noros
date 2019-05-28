#ifndef PROJECTION_PROJECTION_H
#define PROJECTION_PROJECTION_H

#include <Eigen/Core>
#include <Eigen/LU>

#include "camera/matrix.h"


inline Eigen::Vector2f to2d(const Eigen::Vector3f &p) {
    return p.segment(0, 2) / p(2);
}

class Projection {
 public:
    inline Projection(const float fx, const float fy,
                      const float cx, const float cy) :
        K(initializeCameraMatrix(fx, fy, cx, cy)), Kinv(K.inverse()) {
    }

    // u and v are image coordinates so usually ints, but passing in floats for
    // extensibility
    inline Eigen::Vector3f inv_projection(const float u, const float v,
                                          const float depth) const {
        Eigen::Vector3f p;
        p << u, v, 1;
        return depth * Kinv * p;
    }

    const Eigen::Matrix3f K;
    const Eigen::Matrix3f Kinv;
};

#endif /* PROJECTION_PROJECTION_H */
