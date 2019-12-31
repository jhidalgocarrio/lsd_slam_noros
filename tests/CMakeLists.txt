add_executable(
    lsdslam_tests
    depth_estimation/depth_map.cc
    main.cc
)

message("LsdSlam_ALL_LIBRARIES=${LsdSlam_ALL_LIBRARIES}")
set_property(TARGET lsdslam_tests PROPERTY FOLDER "lsd_slam/tests")
target_include_directories(lsdslam_tests PRIVATE ../lsd_slam .)
target_link_libraries(lsdslam_tests
                      ${LsdSlam_ALL_LIBRARIES} ${LIB_CXSPARSE} ${OpenCV_LIBRARIES})

