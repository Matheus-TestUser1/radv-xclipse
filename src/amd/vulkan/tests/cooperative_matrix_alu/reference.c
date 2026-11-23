/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include "../../nir/radv_nir_cmat_muladd_alu.h"
static unsigned
index_of(unsigned column, unsigned r, unsigned c)
{
   return column ? c * 16 + r : r * 16 + c;
}
int
main(void)
{
   unsigned cases = 0, results = 0;
   for (unsigned wave = 32; wave <= 64; wave *= 2)
      for (unsigned bits = 16; bits <= 32; bits *= 2)
         for (unsigned layouts = 0; layouts < 8; layouts++)
            for (unsigned pattern = 0; pattern < 64; pattern++) {
               float A[256], B[256], C[256], ma[256], mb[256], mc[256], out[256] = {0};
               for (unsigned r = 0; r < 16; r++)
                  for (unsigned c = 0; c < 16; c++) {
                     unsigned i = r * 16 + c;
                     A[i] = pattern == 0   ? (r == c ? 1.0f : 0.0f)
                            : pattern == 1 ? 2.0f
                            : pattern == 2 ? 0.0f
                                           : ((int)((r * 3 + c * 5 + pattern * 7) % 17) - 8) * 0.125f;
                     B[i] = pattern == 1   ? 3.0f
                            : pattern == 2 ? 0.0f
                                           : ((int)((r * 7 + c * 3 + pattern * 11) % 19) - 9) * 0.125f;
                     C[i] = pattern == 1 ? 1.0f : (float)(i + 1) * 0.0625f;
                     ma[index_of(layouts & 1, r, c)] = A[i];
                     mb[index_of(layouts & 2, r, c)] = B[i];
                     mc[index_of(layouts & 4, r, c)] = C[i];
                  }
               nir_builder builder = {.wave = wave};
               nir_def *a = alloc_def(&builder, 16, bits), *b = alloc_def(&builder, 16, bits),
                       *acc = alloc_def(&builder, 256 / wave, 32);
               for (unsigned l = 0; l < wave; l++) {
                  for (unsigned k = 0; k < 16; k++) {
                     float av = ma[index_of(layouts & 1, l % 16, k)], bv = mb[index_of(layouts & 2, k, l % 16)];
                     a->value[l][k] = bits == 16 ? half_bits(av) : to_bits(av);
                     b->value[l][k] = bits == 16 ? half_bits(bv) : to_bits(bv);
                  }
                  for (unsigned i = 0; i < 256 / wave; i++)
                     acc->value[l][i] = to_bits(mc[index_of(layouts & 4, l / 16 + i * (wave / 16), l % 16)]);
               }
               nir_def *d = radv_nir_cmat_muladd_alu(&builder, a, b, acc, wave);
               assert(builder.shuffles == (wave == 32 ? 128u : 0u));
               assert(builder.broadcasts == (wave == 64 ? 256u : 0u));
               for (unsigned l = 0; l < wave; l++)
                  for (unsigned i = 0; i < 256 / wave; i++)
                     out[index_of(layouts & 4, l / 16 + i * (wave / 16), l % 16)] = from_bits(d->value[l][i]);
               for (unsigned r = 0; r < 16; r++)
                  for (unsigned c = 0; c < 16; c++) {
                     double expected = C[r * 16 + c];
                     for (unsigned k = 0; k < 16; k++)
                        expected += (double)A[r * 16 + k] * B[k * 16 + c];
                     assert(out[index_of(layouts & 4, r, c)] == (float)expected);
                     results++;
                  }
               for (unsigned i = 0; i < builder.count; i++) {
                  free(builder.allocations[i]);
               }
               cases++;
            }
   printf("PASS driver ALU helper via SIMD primitive adapter: %u cases, %u results, waves32/64, FP16/FP32, eight "
          "layouts; independent dense reference. Wave64: 256 uniform broadcasts, zero variable shuffles. Not a Mesa "
          "build or GPU test.\n",
          cases, results);
}
