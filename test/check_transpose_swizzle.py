"""Run with python3 test/check_transpose_swizzle.py (no GPU required)."""
from kernel_test_support import ROOT, check

source = (ROOT / "src/cl/transpose.cl").read_text().replace('#include "base.cl"', '')
for bits in (32, 64):
    check(source, r'''
int main() {
  std::mt19937_64 random(891);
  std::array<bool, 4096> seen{};
  for (u32 row = 0; row < 64; ++row) {
    for (u32 col = 0; col < 64; ++col) {
      u32 pos = transposeLdsIndex(row, col);
      assert(pos < seen.size() && !seen[pos]);
      seen[pos] = true;
    }
  }
  for (auto [w, h] : {std::pair{64u, 64u}, {256u, 128u}, {128u, 256u}, {256u, 192u}}) {
    std::vector<Word2> input(w * h), output(w * h + 2), restored(w * h);
    for (auto &v : input) { v = {Word(random()), Word(random())}; }
    output.front() = {17, 29}; output.back() = {31, 43};
    const u32 groups = w / 64 * (h / 64);
    for (u32 g = 0; g < groups; ++g) {
      run_group(64, g, groups * 64, [&] {
        transposeWords(w, h, scratch, input.data(), output.data() + 1);
      });
    }
    for (u32 y = 0; y < h; ++y) {
      for (u32 x = 0; x < w; ++x) {
        assert(output[1 + x * h + y] == input[y * w + x]);
      }
    }
    for (u32 g = 0; g < groups; ++g) {
      run_group(64, g, groups * 64, [&] {
        transposeWords(h, w, scratch, output.data() + 1, restored.data());
      });
    }
    assert(input == restored);
    assert((output.front() == Word2{17, 29}));
    assert((output.back() == Word2{31, 43}));
  }
  std::cout << "Transpose permutation, rectangular layouts, both components and round trips passed\n";
}
''', defines=f"#define WordSize {bits // 8}\n#define WIDTH 256\n#define BIG_HEIGHT 128\n",
          cpp_types=f'''
#define restrict __restrict__
using Word = int{bits}_t;
struct Word2 {{ Word x, y; bool operator==(const Word2 &) const = default; }};
{'Word2' if bits == 32 else 'Word'} scratch[4096];
''', cl_types=f"typedef {'int' if bits == 32 else 'long'} Word;\n"
                      f"typedef {'int2' if bits == 32 else 'long2'} Word2;\n")
