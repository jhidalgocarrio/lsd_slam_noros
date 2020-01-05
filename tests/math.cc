#include <math.h>
#include <cmath>
#include <catch.h>
#include <Eigen/Core>
#include "math.h"


TEST_CASE("compute squared cosine angle", "[math]") {
     SECTION("angle is cos(pi/4)") {
         Eigen::Vector2f v1(1, 1);
         Eigen::Vector2f v2(1, 0);
         const float c = cos(M_PI/4);
         REQUIRE(abs(cosine_angle_squared(v1, v2) - c*c) < 1e-6);
     }
     SECTION("angle is cos(0)") {
         Eigen::Vector3f v(1, 1, 1);
         const float c = cos(0);
         REQUIRE(cosine_angle_squared(v, v) == c*c);
     }
     SECTION("angle is cos(pi/2)") {
         Eigen::Vector3f v1(0, 1, 0);
         Eigen::Vector3f v2(1, 0, 1);
         const float c = cos(M_PI/2);
         REQUIRE(abs(cosine_angle_squared(v1, v2) - c*c) < 1e-6);
     }
}
