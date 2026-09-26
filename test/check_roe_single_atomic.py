"""Run with python3 test/check_roe_single_atomic.py (no GPU required)."""
from kernel_test_support import check, excerpt

source = excerpt("src/cl/carryutil.cl", "void updateStats(", "\n#endif")
check(source, r'''
int main() {
  std::mt19937 random(731);
  for (u32 size : {32u, 64u, 128u, 256u}) {
    std::vector<u32> lds(size);
    std::array<u32, STATS_SIZE + 4> stats{};
    stats[STATS_SIZE + 2] = 0xdeadbeef;
    stats[STATS_SIZE + 3] = 0xcafebabe;
    for (u32 sample = 0; sample < STATS_SIZE; ++sample) {
      u32 expected = 0;
      for (u32 g = 0; g < 3; ++g) {
        std::vector<float> values(size);
        for (auto &v : values) {
          v = float(random() % 10000) / 20000;
          expected = std::max(expected, as_uint(v));
        }
        // Ensure every lane can contribute the maximum, including the final lane.
        values[(sample * 17 + size - 1) % size] = 0.75f;
        expected = std::max(expected, as_uint(0.75f));
        run_group(size, g, size * 3, [&] {
          updateStats(lds.data(), size, 3, stats.data(), 0, values[lane]);
        });
        assert(stats[0] == sample + (g == 2));
        assert(stats[1] == (g == 2 ? 0 : g + 1));
      }
      assert(stats[sample + 2] == expected);
    }
    auto full = stats;
    run_group(size, 0, size, [&] {
      updateStats(lds.data(), size, 1, stats.data(), 0, 0.99f);
    });
    assert(stats == full);
  }
  std::cout << "ROE maxima, sample advancement and full-buffer guard passed\n";
}
''', defines="#define STATS_SIZE 8\n")
