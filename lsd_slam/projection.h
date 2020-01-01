
namespace lsd_slam {
    Eigen::Vector2f projection(const Eigen::Vector3f p, const Eigen::Matrix3f &K);
    Eigen::Matrix3f create_intrinsic_matrix(float fx, float fy, float cx, float cy);
}
