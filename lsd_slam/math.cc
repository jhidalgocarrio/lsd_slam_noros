#include <Eigen/Core>
#include "math.h"


const float EPSILON = 1e-16;


float cosine_angle_squared(const Eigen::VectorXf &v1,
                           const Eigen::VectorXf &v2) {
    const float p = v1.dot(v2);
    return (p * p) / (v1.squaredNorm() * v2.squaredNorm());
}


Eigen::VectorXf normalize_length(const Eigen::VectorXf &x) {
    return x / (x.norm() + EPSILON);
}
