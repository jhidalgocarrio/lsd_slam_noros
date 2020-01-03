#include <Eigen/Core>


namespace lsd_slam {

float const EPSILON = 1e-16;

Eigen::Vector2f projection(const Eigen::Vector3f &p) {
    return p.head(2) / (p(2) + EPSILON);
}

Eigen::Vector2f perspective_projection(const Eigen::Vector3f &p,
                                       const Eigen::Matrix3f &K) {
    return projection(K * p);
}

Eigen::Matrix3f create_intrinsic_matrix(float fx, float fy, float cx, float cy) {
    Eigen::Matrix3f K;
    K << fx, 0, cx,
         0, fy, cy,
         0, 0, 1;
    return K;
}

Eigen::Vector3f tohomogeneous(const Eigen::Vector2f &p) {
    const Eigen::Vector3f q(p[0], p[1], 1);
    return q;
}

}
