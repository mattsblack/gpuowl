// Copyright (C) Mihai Preda

#include "base.cl"

// XOR the column with the row to distribute transposed reads across LDS banks.
// Both branches use 8-byte LDS elements. This permutation keeps the 64x64 tile
// at 32 KiB, avoiding the allocation/occupancy increase of a padded stride.
u32 transposeLdsIndex(u32 row, u32 column) {
  return row * 64 + (column ^ row);
}

#if WordSize <= 4

void transposeWords(u32 W, u32 H, local Word2 *lds, global const Word2 *restrict in, global Word2 *restrict out) {
  u32 GPW = W / 64, GPH = H / 64;

  u32 g = get_group_id(0);
  u32 gy = g % GPH;
  u32 gx = g / GPH;
  gx = (gy + gx) % GPW;

  in   += 64 * W * gy + 64 * gx;
  out  += 64 * gy + 64 * H * gx;
  u32 me = get_local_id(0);
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    lds[transposeLdsIndex(i, me)] = in[i * W + me];
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    out[i * H + me] = lds[transposeLdsIndex(me, i)];
  }
}

// from transposed to sequential.
KERNEL(64) transposeOut(P(Word2) out, CP(Word2) in) {
  local Word2 lds[4096];
  transposeWords(WIDTH, BIG_HEIGHT, lds, in, out);
}

// from sequential to transposed.
KERNEL(64) transposeIn(P(Word2) out, CP(Word2) in) {
  local Word2 lds[4096];
  transposeWords(BIG_HEIGHT, WIDTH, lds, in, out);
}

#else

void transposeWords(u32 W, u32 H, local Word *lds, global const Word2 *restrict in, global Word2 *restrict out) {
  u32 GPW = W / 64, GPH = H / 64;

  u32 g = get_group_id(0);
  u32 gy = g % GPH;
  u32 gx = g / GPH;
  gx = (gy + gx) % GPW;

  in   += 64 * W * gy + 64 * gx;
  out  += 64 * gy + 64 * H * gx;
  u32 me = get_local_id(0);
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    lds[transposeLdsIndex(i, me)] = in[i * W + me].x;
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    out[i * H + me].x = lds[transposeLdsIndex(me, i)];
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    lds[transposeLdsIndex(i, me)] = in[i * W + me].y;
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    out[i * H + me].y = lds[transposeLdsIndex(me, i)];
  }
}

// from transposed to sequential.
KERNEL(64) transposeOut(P(Word2) out, CP(Word2) in) {
  local Word lds[4096];
  transposeWords(WIDTH, BIG_HEIGHT, lds, in, out);
}

// from sequential to transposed.
KERNEL(64) transposeIn(P(Word2) out, CP(Word2) in) {
  local Word lds[4096];
  transposeWords(BIG_HEIGHT, WIDTH, lds, in, out);
}

#endif
