#include <Eigen/Core>

inline Eigen::Vector2f projection(const Eigen::Vector3f &p) {
    return p.segment(0, 2) / p(2);
}
