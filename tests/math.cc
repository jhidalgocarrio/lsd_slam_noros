#include <math.h>
#include <cmath>
#include <catch.h>
#include <Eigen/Core>
#include "math.h"


const float THRESHOLD = 1e-6;


TEST_CASE("compute squared cosine angle", "[math]") {
     SECTION("angle is cos(pi/4)") {
         const Eigen::Vector2f v1(1, 1);
         const Eigen::Vector2f v2(1, 0);
         const float c = cos(M_PI/4);
         REQUIRE(abs(cosine_angle_squared(v1, v2) - c*c) < THRESHOLD);
     }
     SECTION("angle is cos(0)") {
         const Eigen::Vector3f v(1, 1, 1);
         const float c = cos(0);
         REQUIRE(cosine_angle_squared(v, v) == c*c);
     }
     SECTION("angle is cos(pi/2)") {
         const Eigen::Vector3f v1(0, 1, 0);
         const Eigen::Vector3f v2(1, 0, 1);
         const float c = cos(M_PI/2);
         REQUIRE(abs(cosine_angle_squared(v1, v2) - c*c) < THRESHOLD);
     }
}


TEST_CASE("normalize vector length", "[math]") {
    SECTION("zero vector") {
        const Eigen::Vector2f u(0, 0);
        const Eigen::Vector2f v = normalize_length(u);
        REQUIRE(v[0] == 0);
        REQUIRE(v[1] == 0);
    }
    SECTION("length is sqrt(5)") {
        const Eigen::Vector2f u(1, 2);
        const Eigen::Vector2f v = normalize_length(u);
        REQUIRE(abs(v[0] - 1/sqrt(5)) < THRESHOLD);
        REQUIRE(abs(v[1] - 2/sqrt(5)) < THRESHOLD);
    }
}
