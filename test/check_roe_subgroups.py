"""Run with python3 test/check_roe_subgroups.py (no GPU required)."""
from kernel_test_support import check, excerpt

source = excerpt("src/cl/carryutil.cl", "void updateStats(", "\n}\n#endif") + "\n}\n"
main = r'''
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
'''


# OpenCL feature macros normally come from the device compiler. The CPU checks
# emulate only collective values, not hardware subgroup scheduling.
emulation = r'''
#ifndef TEST_SUBGROUP_SIZE
#define TEST_SUBGROUP_SIZE 32
#endif
u32 shuffle_values[256];
u32 get_sub_group_id() { return lane / TEST_SUBGROUP_SIZE; }
u32 get_sub_group_local_id() { return lane % TEST_SUBGROUP_SIZE; }
u32 get_num_sub_groups() { return (group_size + TEST_SUBGROUP_SIZE - 1) / TEST_SUBGROUP_SIZE; }
u32 sub_group_reduce_max(u32 v) {
  shuffle_values[lane] = v; bar();
  u32 answer = 0;
  u32 start = get_sub_group_id() * TEST_SUBGROUP_SIZE;
  for (u32 i = start; i < std::min(start + TEST_SUBGROUP_SIZE, group_size); ++i) {
    answer = std::max(answer, shuffle_values[i]);
  }
  bar(); return answer;
}
u32 __shfl_down_sync(u32, u32 v, u32 offset) {
  shuffle_values[lane] = v; bar();
  u32 src = lane % 32 + offset;
  u32 result = src < 32 ? shuffle_values[lane / 32 * 32 + src] : v;
  bar(); return result;
}
'''
for mode, defines in [
    ("fallback", "#undef cl_khr_subgroups\n#undef __opencl_c_subgroups\n"),
    ("OpenCL", "#define cl_khr_subgroups 1\n"),
    ("CUDA", "#define CUDA_BACKEND 1\n")]:
    sizes = [8, 16, 32, 64] if mode == "OpenCL" else [32]
    for size in sizes:
        print(f"Testing {mode}, subgroup size {size}", flush=True)
        check(source, main, defines=defines + f"#define STATS_SIZE 8\n#define TEST_SUBGROUP_SIZE {size}\n",
              cpp_types=emulation,
              cl_types="uint __shfl_down_sync(uint,uint,uint);\n")
