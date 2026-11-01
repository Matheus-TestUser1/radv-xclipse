#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Host checks for FSR 1 selection/constants and real GLSL/SPIR-V compilation.

Run with python3 + cc + c++ + glslangValidator. No GPU or NDK required.
Vulkan metadata is a test double; filtering/constant code is compiled verbatim.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
META = ROOT / "src/amd/vulkan/meta"


def function(source, name):
    match = re.search(r"(?m)^" + re.escape(name) + r"\(", source)
    if not match:
        raise RuntimeError(f"Missing function: {name}")
    start = source.rfind("\n", 0, match.start() - 1) + 1
    opening = source.index("{", match.end())
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end] + "\n"


PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include "vulkan/vulkan_core.h"
#define VK_USE_PLATFORM_ANDROID_KHR
enum { ANDROID_BUFFER_NONE, ANDROID_BUFFER_NATIVE, ANDROID_BUFFER_NATIVE_ALIAS, ANDROID_BUFFER_HARDWARE };
enum { RADV_QUEUE_GENERAL, RADV_QUEUE_COMPUTE, RADV_QUEUE_TRANSFER };
struct test_pool { VkCommandPoolCreateFlags flags; };
struct radv_cmd_buffer {
   struct { VkCommandBufferLevel level; struct test_pool *pool; } vk;
   VkCommandBufferUsageFlags usage_flags;
   int qf;
};
struct radv_image {
   struct {
      VkFormat format; VkExtent3D extent; VkImageType image_type;
      VkSampleCountFlagBits samples; uint32_t mip_levels, array_layers;
      uint64_t create_flags; int android_buffer_type;
   } vk;
   bool wsi_image;
};
'''

TESTS = r'''
int main(void)
{
   struct test_pool pool = {0};
   struct radv_cmd_buffer cmd = {.vk = {VK_COMMAND_BUFFER_LEVEL_PRIMARY, &pool}};
   struct radv_image src = {.vk = {VK_FORMAT_R8G8B8A8_UNORM, {1280, 720, 1}, VK_IMAGE_TYPE_2D,
                                   VK_SAMPLE_COUNT_1_BIT, 1, 1}};
   struct radv_image dst = src;
   dst.vk.extent = (VkExtent3D){1920, 1080, 1}; dst.wsi_image = true;
   VkImageBlit2 region = {.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffsets = {{0, 0, 0}, {1280, 720, 1}}, .dstOffsets = {{0, 0, 0}, {1920, 1080, 1}}};
   VkBlitImageInfo2 info = {.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2, .regionCount = 1,
                           .pRegions = &region, .filter = VK_FILTER_LINEAR};
#define ELIGIBLE(mode) fsr1_eligible(&cmd, &src, &dst, &info, mode)
   assert(!ELIGIBLE(0) && !ELIGIBLE(3) && ELIGIBLE(1) && ELIGIBLE(2));
   dst.wsi_image = false;
   assert(!ELIGIBLE(1) && ELIGIBLE(2));
   dst.vk.android_buffer_type = ANDROID_BUFFER_NATIVE;
   assert(ELIGIBLE(1));
   dst.vk.android_buffer_type = ANDROID_BUFFER_HARDWARE;
   assert(!ELIGIBLE(1));
   dst.wsi_image = true;
   info.filter = VK_FILTER_NEAREST; assert(!ELIGIBLE(1)); info.filter = VK_FILTER_LINEAR;
   info.regionCount = 2; assert(!ELIGIBLE(1)); info.regionCount = 1;
   cmd.qf = RADV_QUEUE_COMPUTE; assert(!ELIGIBLE(1)); cmd.qf = RADV_QUEUE_GENERAL;
   cmd.usage_flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
   assert(!ELIGIBLE(1)); cmd.usage_flags = 0;
   cmd.vk.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY; assert(!ELIGIBLE(1));
   cmd.vk.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   pool.flags = VK_COMMAND_POOL_CREATE_PROTECTED_BIT; assert(!ELIGIBLE(1)); pool.flags = 0;
   src.vk.create_flags = VK_IMAGE_CREATE_PROTECTED_BIT; assert(!ELIGIBLE(1)); src.vk.create_flags = 0;
   src.vk.create_flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT; assert(!ELIGIBLE(1)); src.vk.create_flags = 0;
   src.vk.samples = VK_SAMPLE_COUNT_4_BIT; assert(!ELIGIBLE(1)); src.vk.samples = VK_SAMPLE_COUNT_1_BIT;
   src.vk.mip_levels = 2; assert(!ELIGIBLE(1)); src.vk.mip_levels = 1;
   src.vk.array_layers = 2; assert(!ELIGIBLE(1)); src.vk.array_layers = 1;
   src.vk.image_type = VK_IMAGE_TYPE_3D; assert(!ELIGIBLE(1)); src.vk.image_type = VK_IMAGE_TYPE_2D;
   region.srcOffsets[0].x = 1; assert(!ELIGIBLE(1)); region.srcOffsets[0].x = 0;
   region.dstOffsets[1].x = -1920; assert(!ELIGIBLE(1)); region.dstOffsets[1].x = 1920;
   region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
   assert(!ELIGIBLE(1)); region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   region.dstSubresource.layerCount = VK_REMAINING_ARRAY_LAYERS;
   assert(!ELIGIBLE(1)); region.dstSubresource.layerCount = 1;
   dst.vk.format = VK_FORMAT_B8G8R8A8_UNORM; assert(!ELIGIBLE(1));
   src.vk.format = dst.vk.format; assert(ELIGIBLE(1));
   src.vk.format = dst.vk.format = VK_FORMAT_R8G8B8A8_SRGB; assert(ELIGIBLE(1));
   src.vk.format = dst.vk.format = VK_FORMAT_R16G16B16A16_SFLOAT; assert(!ELIGIBLE(1));
   src.vk.format = dst.vk.format = VK_FORMAT_R8G8B8A8_UNORM;
   dst.vk.extent = src.vk.extent; region.dstOffsets[1] = region.srcOffsets[1]; assert(!ELIGIBLE(1));
   dst.vk.extent = (VkExtent3D){4000, 3000, 1}; region.dstOffsets[1] = (VkOffset3D){4000, 3000, 1};
   assert(!ELIGIBLE(1));
   assert(!fsr1_eligible(&cmd, &src, &src, &info, 2));
   unsetenv("RADV_XCLIPSE_FSR1"); assert(radv_xclipse_fsr1_mode() == 0);
   const char *invalid[] = {"", "-1", "3", "true", "1garbage"};
   for (unsigned i = 0; i < sizeof(invalid)/sizeof(invalid[0]); i++) {
      setenv("RADV_XCLIPSE_FSR1", invalid[i], 1); assert(radv_xclipse_fsr1_mode() == 0);
   }
   setenv("RADV_XCLIPSE_FSR1", "1", 1); assert(radv_xclipse_fsr1_mode() == 1);
   setenv("RADV_XCLIPSE_FSR1", "2", 1); assert(radv_xclipse_fsr1_mode() == 2);
   setenv("RADV_XCLIPSE_FSR1", "0", 1); assert(radv_xclipse_fsr1_mode() == 0);
   return 0;
}
'''

REFERENCE = r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#define A_CPU 1
#include "fsr1/ffx_a.h"
#include "fsr1/ffx_fsr1.h"
#include "radv_fsr1.h"
int main()
{
   const uint32_t cases[][4] = {{1280,720,1920,1080}, {960,540,1920,1080},
      {853,479,1279,719}, {64,64,65,65}, {1920,1080,3840,2160}};
   for (const auto &c : cases) {
      uint32_t ours[20], reference[16];
      radv_fsr1_easu_constants(ours, c[0], c[1], c[2], c[3]);
      FsrEasuCon(reference, reference+4, reference+8, reference+12,
                 float(c[0]), float(c[1]), float(c[0]), float(c[1]), float(c[2]), float(c[3]));
      assert(std::memcmp(ours, reference, sizeof(reference)) == 0);
      assert(ours[16] == c[0] && ours[17] == c[1] && ours[18] == c[2] && ours[19] == c[3]);
   }
   uint32_t rcas[4]; FsrRcasCon(rcas, 1.0f);
   assert(rcas[0] == radv_fsr1_float_bits(0.5f));
}
'''


def main():
    c = (META / "radv_meta_blit.c").read_text()
    config = (ROOT / "src/amd/vulkan/radv_physical_device.c").read_text()
    unit = PRELUDE + function(c, "fsr1_eligible") + function(config, "radv_xclipse_fsr1_mode") + TESTS
    with tempfile.TemporaryDirectory(prefix="radv-fsr1-") as temp:
        temp = Path(temp)
        source = temp / "test.c"
        source.write_text(unit)
        common = ["-I" + str(ROOT / "include"), "-I" + str(META),
                  "-fsanitize=undefined,address", "-fno-sanitize-recover=all"]
        binary = temp / "selection"
        subprocess.run(shlex.split(os.environ.get("CC", "cc")) + common + [
            "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-Wall", "-Wextra", "-Werror",
            "-Wno-missing-field-initializers", "-Wno-sign-compare", str(source), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
        source = temp / "reference.cpp"
        source.write_text(REFERENCE)
        binary = temp / "reference"
        subprocess.run(shlex.split(os.environ.get("CXX", "c++")) + common + [
            "-std=c++11", "-ffp-contract=off", str(source), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
        glslang = shlex.split(os.environ.get("GLSLANG_VALIDATOR", "glslangValidator"))
        for name, stage in (("easu", "comp"), ("rcas", "frag")):
            subprocess.run(glslang + ["-V", "--spirv-val", "-S", stage, "-I" + str(META),
                "-o", str(temp / (name + ".spv")), str(META / ("fsr1_" + name + "." + stage))], check=True)
    print("PASS: FSR1 eligibility, config, AMD reference constants, EASU/RCAS SPIR-V")


if __name__ == "__main__":
    main()
