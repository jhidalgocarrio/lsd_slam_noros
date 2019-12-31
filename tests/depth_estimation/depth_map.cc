#include <vector>

#include <catch.h>

#include "depth_estimation/test_depth_map.h"

using namespace lsd_slam;

TEST_CASE("gradient can be computed", "[gradient]") {
    std::vector<float> intensities{0, 5, 10};
    REQUIRE(calc_grad_along_line(intensities, 1.0) == 50.0);
    REQUIRE(calc_grad_along_line(intensities, 5.0) == 2.0);
}
