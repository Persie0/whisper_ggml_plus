cmake_minimum_required(VERSION 3.10)

project(whisper_flutter_library LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

# Optimization flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fdata-sections -ffunction-sections")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fdata-sections -ffunction-sections")

# Define GGML version and commit (placeholders as we don't have git info here)
add_definitions(-DGGML_VERSION="unknown")
add_definitions(-DGGML_COMMIT="unknown")

# Include directories
# The 'src' folder (relative to this CMakeLists.txt) is android/src/whisper/src
include_directories(src)
include_directories(src/include)
include_directories(src/src)
include_directories(src/ggml/include)
include_directories(src/ggml/src)
include_directories(src/ggml/src/ggml-cpu)
# CoreML is not available on Android
# include_directories(src/src/coreml)

# Collect all engine source files
file(GLOB WHISPER_SRC 
    src/src/*.cpp
    src/src/*.c
    src/ggml/src/*.cpp
    src/ggml/src/*.c
    src/ggml/src/ggml-cpu/*.cpp
    src/ggml/src/ggml-cpu/*.c
    # src/src/coreml/*.cpp
    # src/src/coreml/*.c
    # src/src/coreml/*.m
    # src/src/coreml/*.mm
)

add_library(whisper ${WHISPER_SRC})

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    target_compile_options(whisper PRIVATE -march=armv8.2-a+fp16)
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    target_compile_options(whisper PRIVATE -mfpu=neon-vfpv4)
endif ()

# Main bridge library
add_library(whisper_flutter SHARED main.cpp)
target_link_libraries(whisper_flutter PRIVATE whisper)
target_compile_definitions(whisper_flutter PUBLIC DART_SHARED_LIB)

set_target_properties(whisper_flutter PROPERTIES
  PUBLIC_HEADER src/whisper.h 
  OUTPUT_NAME "whisper"
)
