#include <Eigen/Core>
#include "math.h"

float cosine_angle_squared(const Eigen::VectorXf &v1,
                           const Eigen::VectorXf &v2) {
    const float p = v1.dot(v2);
    return (p * p) / (v1.squaredNorm() * v2.squaredNorm());
}
