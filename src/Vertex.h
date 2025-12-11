#ifndef VERTEX_H
#define VERTEX_H

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <array>


struct Vertex {
    glm::vec3 pos;   // double-precision vec3
    glm::vec3  color; // float vec3

    static VkVertexInputBindingDescription getBindingDescription();
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions();
};

 
#endif
