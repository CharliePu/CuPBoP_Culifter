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

## Region scheduling and scalar-math controls

The incremental [region planner](../src/region_schedule.cpp) analyzes normalized IR before PHI demotion and lane-storage lowering. An eligible single uninterrupted lane region keeps its SSA values and nested-loop PHIs, executing complete lanes sequentially in their original order. It snapshots immutable CTA launch values and statically bounded direct constant-bank loads once per CTA. Pointers loaded from that bank remain pointers; data reached through them is not snapshotted or assumed immutable.

Communication, source allocations, memory-ordering operations, storage requiring ownership lowering and unmodeled helpers retain the established phased path, subject to its existing admission checks. This first separation of scheduling analysis from storage lowering does not implement generalized grid scheduling or a complete rewrite of live state across multiple regions. Physical CTA workers, the host ABI and existing harnesses remain unchanged.

The scalar-math path replaces the registered `rsqrt_f32` call with FP32 `llvm.sqrt` followed by division of 1 by the result. It retains the established CPU wrapper's FP32 sqrt/div value sequence and the caller's source rescaling, with no promise of C `errno` behavior. It introduces no fast math, approximate instructions or reassociation. Other scalar helpers retain their existing contracts.

Both features are enabled by default. For independent ablations, `pipeline.py --no-ssa-regions` selects the legacy phased schedule and `--no-scalar-math` retains the scalar runtime helper. Use both for the legacy scheduling/helper control; use either separately to isolate the changes. The manifest records `region_schedule`, `schedule_reason`, `scalar_math_sites` and both option values. The separate wide-integer control below must also be disabled when reproducing the earlier complete legacy configuration.

The prior SSA/scalar-math evaluation passes 44 legacy/new checks across eleven workloads at one/four workers, with bitwise agreement against legacy and historical outputs. BatchNorm and depthwise convolution use the SSA path; nine workloads retain phased lowering. Focused checks pass 36 semantic executions, seven refusals, eight mixed-scope executions, two tensor cases, 32,768 scalar bit patterns including special values, 12 memory/snapshot cases and four coordinate cases; final compiler IR matches all eleven validated candidates. In that matched four-worker Intel BatchNorm comparison, the combined path takes 128.712447 µs versus legacy 471.003367 µs and oneDNN 20.636416 µs: 3.659× faster than legacy but 6.237× slower than native. Scalar lowering alone regresses; most of the observed gain comes from the structural change. These finite checks do not establish general numerical equivalence or optimal scheduling.

The subsequent [wide-integer normalization](../src/wide_integer.cpp) reconstructs modular full-width arithmetic from low/high words, explicit unsigned carries, sign extension and scaled-word joins. It matches integer relationships, including equivalent low-word expressions, before and between bounded InstCombine cleanup rounds; uncertain patterns remain unchanged. This general normalization can simplify split-word address calculations without adding floating-point, aliasing, signed-overflow or traversal assumptions. `--wide-integers` is enabled by default; `pipeline.py --no-wide-integers` provides the independent ablation. The manifest records `wide_integers` and `wide_integer_sites`; the latter counts rewrite events across cleanup rounds, not unique addresses or dynamic operations.

Independent fixtures pass **6,291,456 modular arithmetic checks** for 8/16/32-bit words, including all 65,536 low-8-bit operand pairs and negative matcher forms. The first fixture try passed arithmetic but failed its 8/16-bit rewrite-coverage assertion because widened output stores hid those patterns; the corrected generator uses width-matched output arrays, and the rejected try is retained. All eleven workloads pass **66 legacy/SSA/wide configurations** at one/four workers with historical outputs bitwise equal; disabling wide normalization reproduces all eleven previous SSA IR files byte-for-byte. Retained semantic, refusal, collective, scalar, snapshot and coordinate regressions pass, with source/IR hashes audited.

Local Core Ultra 9 185H WSL diagnostics at `x86-64-v3` improve BatchNorm over the previous SSA version by **1.211× / 1.154×** at one/four workers; native/new ratios remain **0.240× / 0.232×**. Four-worker native tries vary from 25.35 to 47.38 µs. These diagnostics do not update the earlier Xeon comparison: the server package is prepared, but an existing job triggered the one-node guard and no allocation was requested. Static integer adds fall from 29 to 21 and unsigned comparisons from seven to three; channel loads and traversal remain unchanged.

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
