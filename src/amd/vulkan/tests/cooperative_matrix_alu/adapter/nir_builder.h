/* SPDX-License-Identifier: MIT
 * Eager SIMD adapter for the primitive NIR operations used by the new helper.
 * This is not Mesa NIR and does not claim IR-validation or backend coverage.
 */
#ifndef XCLIPSE_SIMD_NIR_ADAPTER_H
#define XCLIPSE_SIMD_NIR_ADAPTER_H
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
   unsigned num_components, bit_size;
   uint32_t value[64][16];
} nir_def;
typedef struct {
   unsigned wave, count, shuffles, broadcasts;
   nir_def *allocations[4096];
} nir_builder;
static uint32_t
to_bits(float f)
{
   uint32_t u;
   memcpy(&u, &f, 4);
   return u;
}
static float
from_bits(uint32_t u)
{
   float f;
   memcpy(&f, &u, 4);
   return f;
}
static uint16_t
half_bits(float f)
{
   uint32_t u = to_bits(f);
   if (!f)
      return 0;
   return (uint16_t)(((u >> 16) & 0x8000) | ((((u >> 23) & 255) - 112) << 10) | ((u >> 13) & 1023));
}
static float
half_value(uint16_t h)
{
   if (!(h & 0x7fff))
      return 0;
   return from_bits(((uint32_t)(h & 0x8000) << 16) | ((uint32_t)(((h >> 10) & 31) + 112) << 23) |
                    ((uint32_t)(h & 1023) << 13));
}
static nir_def *
alloc_def(nir_builder *b, unsigned n, unsigned bits)
{
   assert(n <= 16 && b->count < 4096);
   nir_def *d = calloc(1, sizeof(*d));
   assert(d);
   d->num_components = n;
   d->bit_size = bits;
   b->allocations[b->count++] = d;
   return d;
}
static nir_def *
nir_channel(nir_builder *b, nir_def *v, unsigned c)
{
   assert(c < v->num_components);
   nir_def *d = alloc_def(b, 1, v->bit_size);
   for (unsigned l = 0; l < b->wave; l++)
      d->value[l][0] = v->value[l][c];
   return d;
}
static nir_def *
nir_f2f32(nir_builder *b, nir_def *v)
{
   assert(v->bit_size == 16);
   nir_def *d = alloc_def(b, 1, 32);
   for (unsigned l = 0; l < b->wave; l++)
      d->value[l][0] = to_bits(half_value((uint16_t)v->value[l][0]));
   return d;
}
static nir_def *
nir_load_subgroup_invocation(nir_builder *b)
{
   nir_def *d = alloc_def(b, 1, 32);
   for (unsigned l = 0; l < b->wave; l++)
      d->value[l][0] = l;
   return d;
}
static nir_def *
nir_ushr_imm(nir_builder *b, nir_def *v, unsigned x)
{
   nir_def *d = alloc_def(b, 1, 32);
   for (unsigned l = 0; l < b->wave; l++)
      d->value[l][0] = v->value[l][0] >> x;
   return d;
}
static nir_def *
nir_iadd_imm(nir_builder *b, nir_def *v, unsigned x)
{
   nir_def *d = alloc_def(b, 1, 32);
   for (unsigned l = 0; l < b->wave; l++)
      d->value[l][0] = v->value[l][0] + x;
   return d;
}
static nir_def *
nir_shuffle(nir_builder *b, nir_def *v, nir_def *index)
{
   b->shuffles++;
   nir_def *d = alloc_def(b, 1, 32);
   for (unsigned l = 0; l < b->wave; l++) {
      assert(index->value[l][0] < b->wave);
      d->value[l][0] = v->value[index->value[l][0]][0];
   }
   return d;
}
static nir_def *
nir_imm_int(nir_builder *b, unsigned v)
{
   nir_def *d = alloc_def(b, 1, 32);
   for (unsigned l = 0; l < b->wave; l++)
      d->value[l][0] = v;
   return d;
}
static nir_def *
nir_ieq_imm(nir_builder *b, nir_def *v, unsigned x)
{
   nir_def *d = alloc_def(b, 1, 1);
   for (unsigned l = 0; l < b->wave; l++)
      d->value[l][0] = v->value[l][0] == x;
   return d;
}
static nir_def *
nir_bcsel(nir_builder *b, nir_def *c, nir_def *yes, nir_def *no)
{
   assert(c->bit_size == 1 && yes->bit_size == no->bit_size);
   nir_def *d = alloc_def(b, 1, yes->bit_size);
   for (unsigned l = 0; l < b->wave; l++)
      d->value[l][0] = c->value[l][0] ? yes->value[l][0] : no->value[l][0];
   return d;
}
static nir_def *
nir_read_invocation(nir_builder *b, nir_def *v, nir_def *idx)
{
   b->broadcasts++;
   for (unsigned l = 0; l < b->wave; l++)
      assert(idx->value[l][0] == idx->value[0][0]);
   assert(idx->value[0][0] < 16);
   nir_def *d = alloc_def(b, 1, 32);
   for (unsigned l = 0; l < b->wave; l++)
      d->value[l][0] = v->value[idx->value[0][0]][0];
   return d;
}
static nir_def *
nir_ffma(nir_builder *b, nir_def *a, nir_def *v, nir_def *c)
{
   nir_def *d = alloc_def(b, 1, 32);
   for (unsigned l = 0; l < b->wave; l++)
      d->value[l][0] = to_bits(fmaf(from_bits(a->value[l][0]), from_bits(v->value[l][0]), from_bits(c->value[l][0])));
   return d;
}
static nir_def *
nir_vec(nir_builder *b, nir_def **v, unsigned n)
{
   nir_def *d = alloc_def(b, n, 32);
   for (unsigned l = 0; l < b->wave; l++)
      for (unsigned i = 0; i < n; i++)
         d->value[l][i] = v[i]->value[l][0];
   return d;
}
#endif
