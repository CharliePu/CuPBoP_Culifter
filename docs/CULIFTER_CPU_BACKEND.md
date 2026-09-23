# CuLifter CPU backend

This fork adds a CPU path for CuLifter-generated LLVM IR while retaining the upstream Vortex tree. The root `build.sh` builds `build/cpu-coarsen` from `src/` and the adapted `cupbop/` passes. It is a separate entry point from the upstream CMake/Vortex pipeline.

## Requirements

Use Linux or WSL, Python 3, and LLVM/Clang 18 development tools. Put `clang++-18`, `clang-18`, `llvm-config-18`, `opt-18` and `llvm-nm-18` on `PATH`. The build uses C++17; `CXX` and `LLVM_CONFIG` can override its compiler and LLVM configuration tool. Pipeline verification/object generation still invokes the versioned tools above.

Keep the [CuLifter repository](https://github.com/CharliePu/CuLifter-multibackends) and [this fork](https://github.com/CharliePu/CuPBoP_Culifter) in sibling directories named `CuLifter` and `CuPBoP`:

```text
workspace/
  CuLifter/test/common/directed_fma.h
  CuPBoP/build.sh
  CuPBoP/pipeline.py
  CuPBoP/runtime/cpu_runtime.cpp
```

The CPU runtime requires the sibling CuLifter `test/common/directed_fma.h` for directed FP32 FMA helpers. Its existing relative include is preserved. That dependency is needed when linking the runtime, separately from building the compiler tool.

## Build and serialize

From the CuPBoP checkout root:

```bash
bash build.sh
python3 pipeline.py INPUT --kernel ENTRY --block-size N \
  -o fresh/cpu.ll --object fresh/cpu.o
```

Replace `INPUT` with the lifted LLVM IR path, `ENTRY` with its selected kernel symbol, and `N` with the exact block volume (1–1024). Use fresh output paths; input IR remains unchanged. The default CPU target is `x86_64-linux-gnu`; `--target aarch64-linux-gnu` selects AArch64 objects, which require a matching runtime/link environment. `--rename` can change the emitted entry symbol. `--shared-memory-bytes` declares the block's shared-memory capacity (default 32768; supported range 1–1048576).

`pipeline.py` performs an incremental build, serializes the selected entry, verifies LLVM IR, and optionally compiles a position-independent CPU object. It writes `cpu.json` with artifact/compiler identities and the launch contract, plus a compiler log. Refused tries write refusal records and retain rejected IR when available; they do not silently fall back to another backend. LLVM verification and object generation do not establish numerical correctness.

The pipeline acquires `/tmp/culifter-benchmark.lock` for its work and refuses another task's lock; it can reuse a lock owned by an ancestor process. Direct `bash build.sh` does not acquire this lock. Coordinate standalone builds and CPU validation accordingly.

## Host runtime contract

The selected kernel must have a zero-argument entry; CuLifter parameters are supplied through its constant-memory ABI. Link the emitted object with [cpu_runtime.cpp](../runtime/cpu_runtime.cpp) and a workload-specific host harness, for example:

```bash
clang++-18 -std=c++17 -O3 -pthread -Iruntime \
  host.cpp runtime/cpu_runtime.cpp fresh/cpu.o -o fresh/run
```

The harness includes [cpu_runtime.hpp](../runtime/cpu_runtime.hpp), declares the emitted symbol as `extern "C" void ENTRY();`, and constructs `cpu_coarsening::Kernel{ENTRY, N}`. Populate `cpu_coarsening::Parameters` with `put` at the original constant-bank offsets, including valid relocated pointers and any required symbol bindings. The parameters contain five 4096-byte banks; zero initialization alone is not a workload binding.

Create a persistent `cpu_coarsening::Executor(workers)` and call `run(kernel, grid, block, parameters)`. The runtime distributes independent CTAs across physical workers; virtual lane/warp loops execute within each CTA. The launch block volume must equal the compiled `N`. An optional affinity vector must contain one valid CPU ID per worker. Kernels using the explicit region-provider hooks also need matching `Regions` callbacks; missing required callbacks raise an error.

Validate complete outputs against the workload's independent reference and fixed numerical contract before accepting execution. Check guards, read-only input preservation and agreement across worker counts. Supply matching shapes, layouts, source arithmetic and input bytes; compilation alone cannot supply this evidence.

## Scope and provenance

This is bounded CTA/warp serialization that preserves lifted computation, live lane state, block-shared state and supported collective ordering. It includes supported full-mask shuffles/barriers and selected tensor forms; acceptance depends on the exact IR and launch contract. Unknown calls, unsupported masks/collectives, nonzero global address spaces, unregistered lane-private globals and unresolved control flow require support or refusal. Inter-CTA synchronization and generalized grid/cluster hierarchy synthesis are outside this contract. It provides no general CUDA, optimal scheduling or generalized HSS guarantee.

The imported CuPBoP_Vortex core is pinned to `ee05a48d79cbae11351e9c8eb711f286c1206402`. Preserve [cupbop/provenance.json](../cupbop/provenance.json), [cupbop/LICENSE](../cupbop/LICENSE), the [repository license](../LICENSE), and upstream source notices. The provenance records upstream identities; locally adapted sources have their own identities in pipeline manifests. `import_upstream.py` is an import/provenance utility, not a build prerequisite; do not run it over local adaptations.
