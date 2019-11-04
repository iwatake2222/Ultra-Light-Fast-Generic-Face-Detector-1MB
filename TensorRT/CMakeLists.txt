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
add_executable(main src/main.cpp /usr/src/tensorrt/samples/common/logger.cpp)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

target_include_directories(main PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    /usr/include/aarch64-linux-gnu/
    /usr/src/tensorrt/samples/common/
)

target_link_libraries(main
    ${OpenCV_LIBS}
    /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so
    /usr/lib/aarch64-linux-gnu/libnvinfer.so
    /usr/lib/aarch64-linux-gnu/libnvonnxparser_runtime.so
    /usr/lib/aarch64-linux-gnu/libnvonnxparser.so
    /usr/lib/aarch64-linux-gnu/libnvparsers.so
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIE}
)


file(COPY ${PROJECT_SOURCE_DIR}/../models/onnx/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)



