cmake_minimum_required(VERSION 3.15)
project(MyProject VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release for single-config generators
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Add executable
add_executable(myapp src/main.cpp)

target_include_directories(myapp PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Compiler warnings
if(MSVC)
  target_compile_options(myapp PRIVATE /W4 /permissive-)
else()
  target_compile_options(myapp PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- GLFW (FetchContent) ----
include(FetchContent)

# Disable GLFW build extras we don't need
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.3.8
)
FetchContent_MakeAvailable(glfw)

# tinyobjloader for model loading
FetchContent_Declare(
  tinyobjloader
  GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
  GIT_TAG v2.0.0
)
FetchContent_MakeAvailable(tinyobjloader)

# Ensure GLFW headers are available to the target (some builds may not
# propagate include dirs automatically depending on how GLFW was added).
if (DEFINED glfw_SOURCE_DIR)
  target_include_directories(myapp PRIVATE ${glfw_SOURCE_DIR}/include)
endif()

# Find Vulkan
find_package(Vulkan REQUIRED)

# Link GLFW and Vulkan to the app
target_link_libraries(myapp PRIVATE glfw Vulkan::Vulkan)

# Link tinyobjloader
target_link_libraries(myapp PRIVATE tinyobjloader)

# Copy a default model next to the executable
set(MODEL_SRC ${CMAKE_SOURCE_DIR}/src/models/model.obj)
if(EXISTS ${MODEL_SRC})
  add_custom_command(TARGET myapp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MODEL_SRC} $<TARGET_FILE_DIR:myapp>
  )
endif()

# Compile GLSL shaders to SPIR-V using glslangValidator from the Vulkan SDK (if available)
find_program(GLSLANG_VALIDATOR_EXECUTABLE glslangValidator HINTS $ENV{VULKAN_SDK}/Bin)
if (GLSLANG_VALIDATOR_EXECUTABLE)
  message(STATUS "Found glslangValidator: ${GLSLANG_VALIDATOR_EXECUTABLE}")
  set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/src/shaders)
  set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
  file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

  set(VERT_SRC ${SHADER_SRC_DIR}/tri.vert)
  set(FRAG_SRC ${SHADER_SRC_DIR}/tri.frag)
  set(VERT_SPV ${SHADER_OUT_DIR}/tri.vert.spv)
  set(FRAG_SPV ${SHADER_OUT_DIR}/tri.frag.spv)

  add_custom_command(
    OUTPUT ${VERT_SPV}
    COMMAND ${GLSLANG_VALIDATOR_EXECUTABLE} -V ${VERT_SRC} -o ${VERT_SPV}
    DEPENDS ${VERT_SRC}
    COMMENT "Compiling ${VERT_SRC} to SPIR-V"
  )

  add_custom_command(
    OUTPUT ${FRAG_SPV}
    COMMAND ${GLSLANG_VALIDATOR_EXECUTABLE} -V ${FRAG_SRC} -o ${FRAG_SPV}
    DEPENDS ${FRAG_SRC}
    COMMENT "Compiling ${FRAG_SRC} to SPIR-V"
  )

  add_custom_target(shaders ALL DEPENDS ${VERT_SPV} ${FRAG_SPV})
  add_dependencies(myapp shaders)

  # Copy SPIR-V files next to the executable after build
  add_custom_command(TARGET myapp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${VERT_SPV} $<TARGET_FILE_DIR:myapp>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FRAG_SPV} $<TARGET_FILE_DIR:myapp>
  )
else()
  message(WARNING "glslangValidator not found; shaders will not be compiled. Install Vulkan SDK or add glslangValidator to PATH.")
endif()

# Install target
install(TARGETS myapp RUNTIME DESTINATION bin)
