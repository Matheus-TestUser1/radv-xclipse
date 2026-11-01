/* SPDX-License-Identifier: MIT */
#version 450
#extension GL_GOOGLE_include_directive : require

#define A_GPU 1
#define A_GLSL 1
#define FSR_RCAS_F 1
#define FSR_RCAS_PASSTHROUGH_ALPHA 1
#include "fsr1/ffx_a.h"
#include "fsr1/ffx_fsr1.h"

layout(set = 0, binding = 0, std430) readonly buffer InputPixels { uint pixels[]; } input_image;
layout(push_constant, std430) uniform Constants {
   uvec4 size; /* output width/height, reserved */
   uvec4 con;
} pc;
layout(location = 0) out vec4 output_color;

AF4 FsrRcasLoadF(ASU2 p)
{
   ivec2 q = clamp(p, ivec2(0), ivec2(pc.size.xy) - 1);
   return unpackUnorm4x8(input_image.pixels[uint(q.y) * pc.size.x + uint(q.x)]);
}
void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) {}

void main()
{
   FsrRcasF(output_color.r, output_color.g, output_color.b, output_color.a,
            uvec2(gl_FragCoord.xy), pc.con);
}
