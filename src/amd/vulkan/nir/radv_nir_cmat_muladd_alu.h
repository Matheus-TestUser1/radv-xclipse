/* SPDX-License-Identifier: MIT */
#ifndef RADV_NIR_CMAT_MULADD_ALU_H
#define RADV_NIR_CMAT_MULADD_ALU_H

#include "nir_builder.h"

/* GFX11 fragment layout, independent of WMMA instruction availability:
 * A[lane][k] = A_matrix[lane % 16][k]
 * B[lane][k] = B_matrix[k][lane % 16]
 * C[lane][i] = C_matrix[lane / 16 + i * wave_size / 16][lane % 16]
 * A/B are replicated in each group of 16 lanes. Read A from lanes 0..15,
 * which are active for a valid subgroup-scope cooperative matrix operation.
 */
static nir_def *
radv_nir_cmat_muladd_alu(nir_builder *b, nir_def *A, nir_def *B, nir_def *C, unsigned wave_size)
{
   assert(wave_size == 32 || wave_size == 64);
   const unsigned length = 256 / wave_size;
   assert(A->num_components == 16 && B->num_components == 16);
   assert((A->bit_size == 16 || A->bit_size == 32) && A->bit_size == B->bit_size);
   assert(C->bit_size == 32 && C->num_components == length);

   nir_def *a[16], *bv[16], *result[8];
   for (unsigned k = 0; k < 16; k++) {
      a[k] = nir_channel(b, A, k);
      bv[k] = nir_channel(b, B, k);
      if (A->bit_size == 16) {
         a[k] = nir_f2f32(b, a[k]);
         bv[k] = nir_f2f32(b, bv[k]);
      }
   }

   nir_def *row_base = nir_ushr_imm(b, nir_load_subgroup_invocation(b), 4);
   nir_def *quarter[4] = {NULL};
   if (wave_size == 64) {
      for (unsigned q = 1; q < 4; q++)
         quarter[q] = nir_ieq_imm(b, row_base, q);
   }
   for (unsigned i = 0; i < length; i++) {
      nir_def *source_lane = wave_size == 32 ? nir_iadd_imm(b, row_base, i * 2) : NULL;
      nir_def *acc = nir_channel(b, C, i);
      for (unsigned k = 0; k < 16; k++) {
         nir_def *av;
         if (wave_size == 64) {
            /* row_base is 0..3. Avoid a divergent full-wave shuffle: on
             * Xclipse/no_permlane64, ACO otherwise enumerates all 64 lanes.
             * Each source index here is subgroup-uniform, so emit_bpermute
             * can use its existing single-readlane path. Select locally
             * without changing EXEC or the cooperative fragment layout.
             * 4 components * 16 K * 4 broadcasts = 256 fixed reads,
             * instead of 4 * 16 * 64 = 4096 generic fallback reads.
             */
            av = nir_read_invocation(b, a[k], nir_imm_int(b, i * 4));
            for (unsigned q = 1; q < 4; q++) {
               nir_def *v = nir_read_invocation(b, a[k], nir_imm_int(b, i * 4 + q));
               av = nir_bcsel(b, quarter[q], v, av);
            }
         } else {
            /* Preserve the already validated wave32 shuffle path. */
            av = nir_shuffle(b, a[k], source_lane);
         }
         acc = nir_ffma(b, av, bv[k], acc);
      }
      result[i] = acc;
   }
   return nir_vec(b, result, length);
}

#endif
