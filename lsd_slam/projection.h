
namespace lsd_slam {
    Eigen::Vector2f perspective_projection(const Eigen::Vector3f &p,
                                           const Eigen::Matrix3f &K);
    Eigen::Vector2f projection(const Eigen::Vector3f &p);
    Eigen::Matrix3f create_intrinsic_matrix(float fx, float fy, float cx, float cy);
    Eigen::Vector3f tohomogeneous(const Eigen::Vector2f &p);
}
