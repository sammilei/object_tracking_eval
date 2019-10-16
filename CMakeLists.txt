cmake_minimum_required(VERSION 3.0.0)
project(DisplayImage VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(DisplayImage main.cpp)
add_executable(Tracker tracker.cpp)
# add_executable(Tracker1 tracker_working.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

find_package(OpenCV REQUIRED PATHS "/home/parallels/opencv_build/opencv/build/opencv2")
include_directories(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(/usr/local/include)


target_link_libraries( DisplayImage ${OpenCV_LIBS} )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( Tracker ${OpenCV_LIBS} )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

# target_link_libraries( Tracker1 ${OpenCV_LIBS} )
# INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")