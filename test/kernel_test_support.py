"""Compile real kernel excerpts and exercise them on CPU work-items.

These checks do not emulate GPU memory ordering, code generation, or performance.
Run the normal device/Gerbicz tests as well before deploying a kernel change.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def excerpt(path, start, end):
    text = (ROOT / path).read_text()
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


CL_PREFIX = r'''
typedef unsigned int u32;
typedef unsigned long u64;
typedef int i32;
typedef long i64;
#define P(T) global T *
#define CP(T) global const T *
#define KERNEL(N) kernel __attribute__((reqd_work_group_size(N, 1, 1))) void
#define OVERLOAD __attribute__((overloadable))
#define assert(X)
void OVERLOAD bar(void) { barrier(CLK_LOCAL_MEM_FENCE); }
void OVERLOAD bar(u32 n) { barrier(CLK_LOCAL_MEM_FENCE); }
'''

CPP_PREFIX = r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;
using uint = unsigned int;
using ulong = unsigned long;
#define P(T) T *
#define CP(T) const T *
#define KERNEL(N) void
#define OVERLOAD
#define global
#define local
#define CLK_LOCAL_MEM_FENCE 1
#define CLK_GLOBAL_MEM_FENCE 2
thread_local u32 lane, group, group_size, global_size;
std::barrier<> *group_barrier;
u32 get_local_id(int) { return lane; }
u32 get_group_id(int) { return group; }
u32 get_local_size(int) { return group_size; }
u32 get_global_id(int) { return group * group_size + lane; }
u32 get_global_size(int) { return global_size; }
void bar() { group_barrier->arrive_and_wait(); }
void bar(u32) { bar(); }
void barrier(int) { bar(); }
void write_mem_fence(int) { std::atomic_thread_fence(std::memory_order_seq_cst); }
u32 as_uint(float f) { return std::bit_cast<u32>(f); }
u32 atomic_add(u32 *p, u32 x) { return std::atomic_ref<u32>(*p).fetch_add(x); }
u32 atomic_max(u32 *p, u32 x) {
  std::atomic_ref<u32> ref(*p);
  u32 old = ref.load();
  while (old < x && !ref.compare_exchange_weak(old, x)) {}
  return old;
}
template<class F> void run_group(u32 size, u32 g, u32 total, F f) {
  std::barrier sync(static_cast<std::ptrdiff_t>(size));
  group_barrier = &sync;
  std::vector<std::jthread> threads;
  for (u32 i = 0; i < size; ++i) {
    threads.emplace_back([=] {
      lane = i; group = g; group_size = size; global_size = total; f();
    });
  }
}
'''


def check(source, main, defines="", cpp_types="", cl_types=""):
    """Syntax-check OpenCL, then run the same excerpt with CPU work-items."""
    clang = os.environ.get("CLANG", shutil.which("clang") or "clang")
    cxx = os.environ.get("KERNEL_TEST_CXX", shutil.which("clang++") or "clang++")
    with tempfile.TemporaryDirectory(prefix="prpll-kernel-test-") as tmp:
        path = Path(tmp)
        (path / "check.cl").write_text(CL_PREFIX + defines + cl_types + source)
        subprocess.run([clang, "-target", "spir64", "-x", "cl", "-cl-std=CL2.0", "-fsyntax-only",
                        str(path / "check.cl")], check=True)
        # Share kernel-local arrays between CPU work-items. Groups execute in
        # sequence; caller-supplied LDS parameters remain ordinary pointers.
        host_source = re.sub(r"\blocal (\w+) (\w+)\[", r"static \1 \2[", source)
        (path / "check.cpp").write_text(CPP_PREFIX + defines + cpp_types + host_source + main)
        subprocess.run([cxx, "-std=c++20", "-O2", "-pthread", "-Wno-unknown-pragmas",
                        "-fsanitize=undefined", str(path / "check.cpp"),
                        "-o", str(path / "check")], check=True)
        subprocess.run([str(path / "check")], check=True, timeout=120)
