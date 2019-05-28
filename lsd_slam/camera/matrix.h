#include <Eigen/Core>


inline Eigen::Matrix3f initializeCameraMatrix(const float fx, const float fy,
                                              const float cx, const float cy) {
    Eigen::Matrix3f K;
    K << fx, 0, cx,
         0, fy, cy,
         0, 0, 1;
    return K;
}

