#include <vector>

#include <Eigen/Dense>
#include <catch.h>

#include "depth_estimation/test_depth_map.h"

using namespace lsd_slam;

TEST_CASE("gradient can be computed", "[gradient]") {
    SECTION("intensity pattern 1") {
        Eigen::VectorXf intensities(3);
        intensities << 0, 5, 10;
        REQUIRE(calc_grad_along_line(intensities, 1.0) == 50.0);  // 50 / (1 * 1)
        REQUIRE(calc_grad_along_line(intensities, 5.0) == 2.0);   // 50 / (5 * 5)
    }

    SECTION("intensity pattern 2") {
        Eigen::VectorXf intensities(3);
        intensities << 0, 1, 4;
        REQUIRE(calc_grad_along_line(intensities, 1.0) == 10.0);  // 10 / (1 * 1)
        REQUIRE(calc_grad_along_line(intensities, 2.0) == 2.5);  // 10 / (2 * 2)
    }
}
