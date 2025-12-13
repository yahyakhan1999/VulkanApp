cmake_minimum_required(VERSION 3.10)
project(MyVulkanApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find GLFW package
find_package(glfw3 3.3 REQUIRED)

# Find Vulkan package
find_package(Vulkan REQUIRED)

add_executable(MyVulkanApp
    src/main.cpp
    src/Vertex.cpp
    # other source files
)

# Link Vulkan and GLFW libraries
target_link_libraries(MyVulkanApp PRIVATE glfw Vulkan::Vulkan)

# Add include directory for stb_image.h
target_include_directories(MyVulkanApp PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Optionally, add compiler warnings or defines:
# target_compile_options(MyVulkanApp PRIVATE -Wall -Wextra -Wpedantic)
