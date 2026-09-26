"""Run with python3 test/check_gf61_square.py (no GPU required)."""
from kernel_test_support import check, excerpt

source = excerpt("src/cl/math.cl", "u128 square64(", "\nu128 OVERLOAD mad64(")
source += excerpt("src/cl/math.cl", "u64 OVERLOAD weakModM61(", "\n// Returns a * b")
source += excerpt("src/cl/math.cl", "Z61 weakSquare(", "\nZ61 OVERLOAD weakMulAdd(")
types = r'''
typedef u64 Z61;
typedef struct { u64 hi64; u64 lo64; } u128;
u128 make_u128(u64 hi, u64 lo) { u128 v; v.hi64 = hi; v.lo64 = lo; return v; }
u64 u128_lo64(u128 a) { return a.lo64; }
u64 u128_hi64(u128 a) { return a.hi64; }
'''
check(source, r'''
void checkSquare(u64 a) {
  unsigned __int128 expected = (unsigned __int128) a * a;
  u128 actual = square64(a);
  assert(u128_lo64(actual) == u64(expected));
  assert(u128_hi64(actual) == u64(expected >> 64));
  // A count of 9 permits the entire 64-bit input range (128-bit reduction).
  assert(weakSquare(a, 9) % M61 == expected % M61);
}
int main() {
  for (u64 a : {0ULL, 1ULL, 0xffffffffULL, 0x100000000ULL,
                0x1fffffffffffffffULL, 0x2000000000000000ULL,
                0x7fffffffffffffffULL, 0x8000000000000000ULL,
                0xffffffff00000000ULL, 0xffffffffffffffffULL}) {
    checkSquare(a);
  }
  for (u32 bit = 0; bit < 64; ++bit) {
    u64 p = u64(1) << bit;
    checkSquare(p); checkSquare(p - 1); checkSquare(p + 1);
  }
  std::mt19937_64 random(881);
  for (u32 i = 0; i < 100000; ++i) {
    checkSquare(random());
    for (u32 count : {2u, 3u, 4u}) {
      u64 a = random() % ((count - 1) * M61 + 1);
      unsigned __int128 expected = (unsigned __int128) a * a;
      u64 actual = weakSquare(a, count);
      u128 full = make_u128(u64(expected >> 64), u64(expected));
      // Require the same weak representation, not just modular equivalence.
      assert(actual == weakModM61(full, count <= 3 ? 125 : 128));
      assert(actual % M61 == expected % M61);
    }
  }
  std::cout << "Full-width squares, cross-product carries and M61 reductions passed\n";
}
''', defines="#define HAS_PTX 0\n#define M61 0x1fffffffffffffffULL\n",
      cpp_types=types, cl_types=types)
