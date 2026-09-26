"""Run with python3 test/check_checksum_reduction.py (no GPU required)."""
from kernel_test_support import check, excerpt

source = excerpt("src/cl/etc.cl", "KERNEL(64) sum64(", "\n#endif")
for bits in (32, 64):
    check(source, r'''
int main() {
  std::mt19937_64 random(821);
  for (u32 count : {0u, 1u, 31u, 63u, 64u, 65u, 127u, 513u, 4099u}) {
    for (u32 pattern = 0; pattern < 3; ++pattern) {
      std::vector<Word> input(count);
      for (auto &v : input) {
        v = pattern == 0 ? Word(-1) : pattern == 1 ? std::numeric_limits<Word>::min() : Word(random());
      }
      // Exercise low-word overflow and nonzero initial output as well.
      u64 expected = 0x12345678ffffffffULL;
      alignas(8) u32 output[2] = {u32(expected), u32(expected >> 32)};
      for (auto v : input) { expected += u64(v); }
      for (u32 g = 0; g < 4; ++g) {
        run_group(64, g, 256, [&] {
          sum64(reinterpret_cast<ulong *>(output), count, input.data());
        });
      }
      u64 actual = (u64(output[1]) << 32) | output[0];
      assert(actual == expected);
    }
  }
  std::cout << "Checksum signed inputs, tails, empty input and wraparound passed\n";
}
''', cpp_types=f"#include <limits>\nusing Word = int{bits}_t;\n",
          cl_types=f"typedef {'int' if bits == 32 else 'long'} Word;\n")
