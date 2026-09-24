#pragma once
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

namespace cpu_schedule {
struct LaneIndices {
  llvm::AllocaInst *intra = nullptr, *inter = nullptr;
  unsigned count() const { return (intra != nullptr) + (inter != nullptr); }
};
// CuPBoP's lane/warp loop indices are thread-local runtime globals. Kernel
// memory accesses through unknown pointers may alias them, so LLVM reloads the
// index after every store and cannot form a canonical induction variable.
// When the kernel is the only function accessing an index, and only through
// direct loads/stores, keep the index in a function-local slot (promoted to
// SSA by LLVM). The TLS object is loaded at entry and written back before
// every non-intrinsic call and every return, and reloaded after each such
// call, so every observer of the TLS object sees the same values as before.
LaneIndices localizeLaneIndices(llvm::Function &kernel);

// CuPBoP keeps per-lane predicates (masks, pending flags, saved compares) in
// i1 lane arrays. LLVM's vectorizer scalarizes every access to such an
// irregular type. Store them as bytes instead (zext on store, trunc on load);
// the stored predicate values are unchanged. Returns the number of arrays.
unsigned widenLaneMasks(llvm::Function &kernel);

struct ParallelLaneLoops {
  unsigned laneLoops = 0, asserted = 0, forced = 0;
};
// insert_warp_loop proposes llvm.loop.parallel_accesses on every lane loop.
// Keep the proposal only where this structural check proves that no memory
// dependence crosses lanes; elsewhere remove it. Proved loops tag their
// accesses with !llvm.access.group; with forceVectorize they also request
// llvm.loop.vectorize.enable (never with sub-byte lane storage). Each decision
// is reported on stderr.
ParallelLaneLoops annotateParallelLaneLoops(llvm::Function &kernel,
                                            const LaneIndices &indices,
                                            bool prove = true,
                                            bool forceVectorize = false);
}
