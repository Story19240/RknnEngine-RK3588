cmake_minimum_required(VERSION 3.22.1)

project(rknnengine LANGUAGES CXX)

if(NOT ANDROID)
    message(FATAL_ERROR "rknnengine is an Android native library")
endif()

if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
    message(FATAL_ERROR "rknnengine supports arm64-v8a only")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OpenCV_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv-mobile-4.10.0-android/sdk/native/jni"
        CACHE PATH "OpenCV mobile package root" FORCE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_library(rknnrt SHARED IMPORTED)
set_target_properties(rknnrt PROPERTIES
        IMPORTED_LOCATION
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn/arm64-v8a/librknnrt.so")

add_library(rknnengine SHARED
        jni/rknn_engine_jni.cpp
        rknn/rknn_wrapper.cpp
        rknn/rknn_yolov8_output_processor.cpp)

target_include_directories(rknnengine PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn")

target_compile_features(rknnengine PRIVATE cxx_std_17)
target_compile_options(rknnengine PRIVATE
        -Wall
        -Wextra
        -Wpedantic)

target_link_libraries(rknnengine
        PRIVATE
        rknnrt
        ${OpenCV_LIBS}
        android
        jnigraphics
        log)
