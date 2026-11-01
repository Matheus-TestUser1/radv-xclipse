/*
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Scalar C form of FsrEasuCon from fsr1/ffx_fsr1.h. See that file's MIT notice.
 */
#ifndef RADV_FSR1_H
#define RADV_FSR1_H

#include <stdint.h>
#include <string.h>

static inline uint32_t
radv_fsr1_float_bits(float value)
{
   uint32_t bits;
   memcpy(&bits, &value, sizeof(bits));
   return bits;
}

static inline void
radv_fsr1_easu_constants(uint32_t constants[20], uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh)
{
   const float ix = 1.0f / (float)sw, iy = 1.0f / (float)sh;
   const float sx = (float)sw * (1.0f / (float)dw), sy = (float)sh * (1.0f / (float)dh);
   const float values[16] = {
      sx, sy, 0.5f * sx - 0.5f, 0.5f * sy - 0.5f,
      ix, iy, ix, -iy,
      -ix, 2.0f * iy, ix, 2.0f * iy,
      0.0f, 4.0f * iy, 0.0f, 0.0f,
   };
   for (unsigned i = 0; i < 16; i++)
      constants[i] = radv_fsr1_float_bits(values[i]);
   constants[16] = sw;
   constants[17] = sh;
   constants[18] = dw;
   constants[19] = dh;
}

#endif
