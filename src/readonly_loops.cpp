#include "readonly_loops.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

using namespace llvm;
namespace cpu_schedule {
namespace {
bool readOnlyLoop(const Loop &L) {
  for (auto *B : L.blocks())
    for (auto &I : *B) {
      if (isa<AllocaInst>(I) || I.isAtomic() || isa<FenceInst>(I) ||
          I.mayWriteToMemory())
        return false;
      if (auto *Load = dyn_cast<LoadInst>(&I))
        if (Load->isVolatile()) return false;
      // In particular, reject collectives, coordinate-observing runtime
      // helpers, constrained FP operations, and unmodeled external calls.
      if (auto *Call = dyn_cast<CallBase>(&I))
        if (!isa<IntrinsicInst>(Call) || !Call->doesNotAccessMemory() ||
            Call->isConvergent() || Call->mayHaveSideEffects())
          return false;
    }
  return true;
}

void gather(Loop &L, SmallVectorImpl<Loop *> &Candidates) {
  if (readOnlyLoop(L)) {
    Candidates.push_back(&L);
    return;
  }
  for (auto *Child : L.getSubLoops()) gather(*Child, Candidates);
}
} // namespace

unsigned outlineReadOnlyLoops(Function &F) {
  unsigned Count = 0;
  // Extraction changes both CFG and loop ownership. Recompute analyses and
  // extract one maximal eligible region per round, in source loop order.
  while (true) {
    DominatorTree DT(F);
    LoopInfo LI(DT);
    SmallVector<Loop *, 8> Candidates;
    for (auto *L : LI) gather(*L, Candidates);
    bool Changed = false;
    for (auto *L : Candidates) {
      CodeExtractor Extractor(DT, *L, false, nullptr, nullptr, nullptr,
                              "cpu.readonly.loop." + std::to_string(Count));
      if (!Extractor.isEligible()) continue;
      CodeExtractorAnalysisCache Cache(F);
      auto *Helper = Extractor.extractCodeRegion(Cache);
      if (!Helper) continue;
      Helper->removeFnAttr("cpu.coarsen.kernel");
      Helper->removeFnAttr("cpu.block_size");
      Helper->addFnAttr("cpu.readonly.loop");
      // The outlined body is already a complete single-lane operation. Its
      // SSA loop PHIs must not be demoted or scheduled as another kernel.
      auto *Call = cast<CallInst>(*Helper->user_begin());
      auto *Block = Call->getParent();
      auto Barrier = F.getParent()->getOrInsertFunction(
          "llvm.nvvm.barrier0", Type::getVoidTy(F.getContext()));
      // Preserve the mask scheduler's ordering with surrounding, possibly
      // aliasing writes. Read-only proves iteration interchange inside this
      // region, not that entry/exit boundaries can be removed. These are real
      // cuts for serialization, not removable synthetic CFG separators.
      IRBuilder<> Before(&*Block->getFirstInsertionPt());
      Before.CreateCall(Barrier);
      IRBuilder<> After(Block->getTerminator());
      After.CreateCall(Barrier);
      ++Count;
      Changed = true;
      break;
    }
    if (!Changed) break;
  }
  F.addFnAttr("cpu.readonly.loop.regions", std::to_string(Count));
  return Count;
}
} // namespace cpu_schedule
