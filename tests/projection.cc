#include <Eigen/Core>
#include <catch.h>
#include "projection.h"

using namespace lsd_slam;

TEST_CASE("compute homogeneous form", "[projection]") {
    Eigen::Vector2f p(2, 4);
    Eigen::Vector3f q = tohomogeneous(p);
    assert(q(0) == 2);
    assert(q(1) == 4);
    assert(q(2) == 1);
}


TEST_CASE("intrinsic matrix can be created", "[projection]") {
    Eigen::Matrix3f K = create_intrinsic_matrix(10, 12, 3, 4);
    REQUIRE(K(0, 0) == 10);
    REQUIRE(K(0, 1) == 0);
    REQUIRE(K(0, 2) == 3);
    REQUIRE(K(1, 0) == 0);
    REQUIRE(K(1, 1) == 12);
    REQUIRE(K(1, 2) == 4);
    REQUIRE(K(2, 0) == 0);
    REQUIRE(K(2, 1) == 0);
    REQUIRE(K(2, 2) == 1);
}

TEST_CASE("perspective projection", "[projection]") {
    Eigen::Vector3f p(10, 20, 5);
    SECTION("intrinsic is identity") {
        Eigen::Matrix3f K;
        K << 1, 0, 0,
             0, 1, 0,
             0, 0, 1;
        Eigen::Vector2f q = perspective_projection(p, K);
        REQUIRE(q(0) == 2);
        REQUIRE(q(1) == 4);
    }

    SECTION("intrinsic is not identity") {
        Eigen::Matrix3f K;
        K << 3, 0, 4,
             0, 4, 2,
             0, 0, 1;
        Eigen::Vector2f q = perspective_projection(p, K);
        REQUIRE(q(0) == 10);
        REQUIRE(q(1) == 18);
    }
}

TEST_CASE("division by z ", "[projection]") {
    Eigen::Vector3f p(10, 20, 5);
    Eigen::Vector2f q = projection(p);
    REQUIRE(q(0) == 2);
    REQUIRE(q(1) == 4);
}
