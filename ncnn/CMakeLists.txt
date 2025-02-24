cmake_minimum_required(VERSION 3.10)

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "No build type selected, default to Release")
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
    endif()
endif()

project(Ultra-Light-Fast-Generic-Face-Detector-1MB)

add_subdirectory(3rdparty/ncnn ncnn_build)
include_directories(3rdparty/ncnn/src)

find_package(OpenCV REQUIRED)

# add_executable(main src/main.cpp src/UltraFace.cpp)
add_executable(main src/main_camera.cpp src/UltraFace.cpp)

target_link_libraries(main ncnn ${OpenCV_LIBS})
target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})

file(COPY ${PROJECT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)



## note
# Build for Raspberry Pi
# cmake -DCMAKE_TOOLCHAIN_FILE=3rdparty/ncnn/toolchains/pi3.toolchain.cmake -DPI3=ON ..
