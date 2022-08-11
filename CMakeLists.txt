cmake_minimum_required(VERSION 3.0.2)
project(vehicle_control)

find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
    sensor_msgs
)

catkin_python_setup()

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
        vehicle_control_main.py
        config.json
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

install(DIRECTORY
        launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )