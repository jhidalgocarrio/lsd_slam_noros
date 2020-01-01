#include <Eigen/Core>


namespace lsd_slam {

float const EPSILON = 1e-16;

Eigen::Vector2f projection(const Eigen::Vector3f p, const Eigen::Matrix3f &K) {
    Eigen::Vector3f q = K * p;
    return q.head(2) / (q(2) + EPSILON);
}

Eigen::Matrix3f create_intrinsic_matrix(float fx, float fy, float cx, float cy) {
    Eigen::Matrix3f K;
    K << fx, 0, cx,
         0, fy, cy,
         0, 0, 1;
    return K;
}

}
