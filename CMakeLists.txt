cmake_minimum_required(VERSION 3.8)
project(pid_maze_solver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(pid_maze_solver src/pid_maze_solver.cpp)
ament_target_dependencies(pid_maze_solver rclcpp std_msgs geometry_msgs nav_msgs tf2 Eigen3 yaml-cpp sensor_msgs ament_index_cpp)
target_link_libraries(pid_maze_solver ${YAML_CPP_LIBRARIES})

install(TARGETS
  pid_maze_solver
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY
    waypoints
  DESTINATION
    share/${PROJECT_NAME}/
)

ament_package()
