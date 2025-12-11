#!/bin/bash

# Compile vertex shader
glslangValidator -V graphicsShader.vert -o graphicsShader.vert.spv
if [ $? -ne 0 ]; then
  echo "Vertex shader compilation failed"
  exit 1
fi
echo "Vertex shader compiled successfully to graphicsShader.vert.spv"

# Compile fragment shader
glslangValidator -V graphicsShader.frag -o graphicsShader.frag.spv
if [ $? -ne 0 ]; then
  echo "Fragment shader compilation failed"
  exit 1
fi
echo "Fragment shader compiled successfully to graphicsShader.frag.spv"
