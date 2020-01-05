#include <vector>

#include <Eigen/Dense>
#include <catch.h>

#include "depth_estimation/depth_map.h"


TEST_CASE("computing intensity gradient", "[gradient]") {
    SECTION("intensity pattern 1") {
        const Eigen::Vector3f intensities(0, 5, 10);
        REQUIRE(calc_grad_along_line(intensities, 1.0) == 50.0);  // 50 / (1 * 1)
        REQUIRE(calc_grad_along_line(intensities, 5.0) == 2.0);   // 50 / (5 * 5)
    }

    SECTION("intensity pattern 2") {
        const Eigen::Vector3f intensities(0, 1, 4);
        REQUIRE(calc_grad_along_line(intensities, 1.0) == 10.0);  // 10 / (1 * 1)
        REQUIRE(calc_grad_along_line(intensities, 2.0) == 2.5);  // 10 / (2 * 2)
    }
}


TEST_CASE("check if given coordinate is in the image range", "[image]") {
    const Eigen::Vector2i image_size(200, 300);

    SECTION("padding is 0") {
        const Eigen::Vector2f p1(-1, 0);
        REQUIRE(not is_in_image_range(p1, image_size));
        const Eigen::Vector2f p2(0, -1);
        REQUIRE(not is_in_image_range(p2, image_size));
        const Eigen::Vector2f p3(0, 0);
        REQUIRE(is_in_image_range(p3, image_size));
        const Eigen::Vector2f p4(199, 299);
        REQUIRE(is_in_image_range(p4, image_size));
        const Eigen::Vector2f p5(200, 299);
        REQUIRE(not is_in_image_range(p5, image_size));
        const Eigen::Vector2f p6(199, 300);
        REQUIRE(not is_in_image_range(p6, image_size));
    }

    SECTION("padding is 1") {
        const Eigen::Vector2f p1(1, 0);
        REQUIRE(not is_in_image_range(p1, image_size, 1));
        const Eigen::Vector2f p2(0, 1);
        REQUIRE(not is_in_image_range(p2, image_size, 1));
        const Eigen::Vector2f p3(1, 1);
        REQUIRE(is_in_image_range(p3, image_size, 1));
        const Eigen::Vector2f p4(198, 298);
        REQUIRE(is_in_image_range(p4, image_size, 1));
        const Eigen::Vector2f p5(199, 298);
        REQUIRE(not is_in_image_range(p5, image_size, 1));
        const Eigen::Vector2f p6(198, 299);
        REQUIRE(not is_in_image_range(p6, image_size, 1));
    }
}
