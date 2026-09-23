#include "region_schedule.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include <map>
#include <vector>

using namespace llvm;
namespace cpu_schedule {
static bool uniformCoordinate(StringRef N) {
  for (StringRef Known : {"block_size", "block_size_x", "block_size_y", "block_size_z",
                         "block_index_x", "block_index_y", "block_index_z",
                         "grid_size_x", "grid_size_y", "grid_size_z"})
    if (N == Known) return true;
  return false;
}

unsigned lowerScalarMath(Function &F) {
  // GPU scalar arithmetic has no C errno side effect. Preserve the CPU
  // backend's established 1/sqrt FP32 value sequence, without fast-math,
  // approximate CPU instructions, or reassociation. Source rescaling remains
  // in the caller. Other helpers retain their existing runtime contracts.
  std::vector<CallInst *> calls;
  for (auto &I : instructions(F))
    if (auto *C = dyn_cast<CallInst>(&I))
      if (auto *CF = C->getCalledFunction())
        if (CF->isDeclaration() && CF->getName() == "rsqrt_f32" &&
            C->arg_size() == 1 && C->getType()->isFloatTy() &&
            C->getArgOperand(0)->getType()->isFloatTy())
          calls.push_back(C);
  for (auto *C : calls) {
    IRBuilder<> B(C);
    auto *Sqrt = Intrinsic::getDeclaration(F.getParent(), Intrinsic::sqrt,
                                          {B.getFloatTy()});
    auto *Root = B.CreateCall(Sqrt, {C->getArgOperand(0)}, "cpu.sqrt");
    auto *Reciprocal = B.CreateFDiv(ConstantFP::get(B.getFloatTy(), 1.0), Root,
                                   "cpu.rsqrt");
    C->replaceAllUsesWith(Reciprocal);
    C->eraseFromParent();
  }
  return calls.size();
}

Plan analyze(const Function &F) {
  for (const auto &I : instructions(F)) {
    if (isa<AllocaInst>(I)) return {Kind::Phased, "source allocation requires lane lifetime handling"};
    if (I.isAtomic() || isa<FenceInst>(I))
      return {Kind::Phased, "atomic or memory ordering operation"};
    const Value *P = nullptr;
    if (auto *L = dyn_cast<LoadInst>(&I)) {
      if (L->isVolatile()) return {Kind::Phased, "volatile load"};
      P = L->getPointerOperand();
    }
    if (auto *S = dyn_cast<StoreInst>(&I)) {
      if (S->isVolatile()) return {Kind::Phased, "volatile store"};
      P = S->getPointerOperand();
      // The source cannot modify execution coordinates through this path.
      if (auto *G = dyn_cast<GlobalVariable>(getUnderlyingObject(P)))
        if (G->isThreadLocal()) return {Kind::Phased, "source writes runtime state"};
    }
    if (P)
      if (auto *G = dyn_cast<GlobalVariable>(getUnderlyingObject(P)))
        if (G->getName() == "shared_mem" || G->getName() == "local_mem" ||
            G->getName() == "cpu_shared_memory" || G->getName() == "cpu_local_memory")
          return {Kind::Phased, "shared or lane-private memory requires ownership lowering"};
    if (auto *C = dyn_cast<CallBase>(&I)) {
      auto *CF = C->getCalledFunction();
      auto N = CF ? CF->getName() : StringRef();
      // These registered runtime functions do not observe or mutate logical
      // coordinates. This scheduling fact does NOT assert memory(none): the
      // retained runtime call still has its conservative LLVM effect model.
      bool scalarRuntime = N == "rsqrt_f32" || N == "rcp_f32" ||
          N == "fma_rm_f32" || N == "fma_rp_f32" || N == "fma_rz_f32";
      if (!scalarRuntime && (!isa<IntrinsicInst>(C) || !C->doesNotAccessMemory() || C->isConvergent()))
        return {Kind::Phased, "communication or unmodeled helper requires phased lowering"};
    }
  }
  if (F.hasFnAttribute("cpu.generic.shared.pointer"))
    return {Kind::Phased, "generic shared pointer"};
  return {Kind::WholeLane, "one uninterrupted lane region; preserve SSA and sequential lane order"};
}

void emitWholeLane(Function &F, unsigned BlockSize) {
  auto &C = F.getContext();
  auto &M = *F.getParent();
  auto *Body = &F.getEntryBlock();
  // Entry blocks have no incoming edges before the transformation; all PHIs
  // in ordinary nested loops retain their original incoming edge/value pairs.
  auto *Entry = BasicBlock::Create(C, "cpu.dispatch", &F, Body);
  auto *Header = BasicBlock::Create(C, "cpu.lane.header", &F, Body);
  auto *Latch = BasicBlock::Create(C, "cpu.lane.latch", &F);
  auto *Exit = BasicBlock::Create(C, "cpu.dispatch.exit", &F);
  IRBuilder<> B(Entry);
  B.CreateBr(Header);
  B.SetInsertPoint(Header);
  auto *Lane = B.CreatePHI(B.getInt32Ty(), 2, "cpu.lane");
  Lane->addIncoming(B.getInt32(0), Entry);
  B.CreateBr(Body);

  std::vector<LoadInst *> coordinates, uniformCoordinates, constantLoads;
  std::vector<ReturnInst *> returns;
  for (auto &I : instructions(F)) {
    if (auto *L = dyn_cast<LoadInst>(&I)) {
      if (L->getPointerOperand() == M.getNamedGlobal("intra_warp_index") ||
          L->getPointerOperand() == M.getNamedGlobal("inter_warp_index"))
        coordinates.push_back(L);
      else if (auto *G = dyn_cast<GlobalVariable>(L->getPointerOperand()))
        if (uniformCoordinate(G->getName()))
          uniformCoordinates.push_back(L);
      int64_t Offset = 0;
      auto *Base = GetPointerBaseWithConstantOffset(L->getPointerOperand(), Offset, M.getDataLayout());
      auto *Bank = M.getNamedGlobal("const_mem");
      if (Bank && Base == Bank && !L->isVolatile() && !L->isAtomic()) {
        auto Bytes = M.getDataLayout().getTypeStoreSize(L->getType());
        auto Capacity = M.getDataLayout().getTypeAllocSize(Bank->getValueType());
        // The parameter/constant bank is immutable for an active CTA. Only
        // statically bounded direct loads are eagerly evaluated; data loads
        // through pointers stored in this bank are deliberately NOT hoisted.
        if (!Bytes.isScalable() && !Capacity.isScalable() && Offset >= 0 &&
            uint64_t(Offset) <= Capacity.getFixedValue() &&
            Bytes.getFixedValue() <= Capacity.getFixedValue() - uint64_t(Offset))
          constantLoads.push_back(L);
      }
    }
    if (auto *R = dyn_cast<ReturnInst>(&I)) returns.push_back(R);
  }
  for (auto *L : coordinates) {
    Value *V = L->getPointerOperand() == M.getNamedGlobal("intra_warp_index")
                   ? static_cast<Value *>(Lane) : B.getInt32(0);
    L->replaceAllUsesWith(V);
    L->eraseFromParent();
  }
  // GPU special registers are immutable within a CTA. Materialize their ABI
  // values once, independently of alias uncertainty in ordinary data buffers.
  std::map<GlobalVariable *, Value *> snapshots;
  IRBuilder<> Uniform(Entry->getTerminator());
  for (auto *L : uniformCoordinates) {
    auto *G = cast<GlobalVariable>(L->getPointerOperand());
    if (!snapshots.count(G))
      snapshots[G] = G->getName() == "block_size"
          ? static_cast<Value *>(Uniform.getInt32(BlockSize))
          : Uniform.CreateLoad(G->getValueType(), G, G->getName() + ".cta");
    L->replaceAllUsesWith(snapshots[G]);
    L->eraseFromParent();
  }
  for (auto *L : constantLoads) {
    int64_t Offset = 0;
    auto *Base = GetPointerBaseWithConstantOffset(L->getPointerOperand(), Offset, M.getDataLayout());
    auto *Address = Uniform.CreateGEP(Uniform.getInt8Ty(), Base, Uniform.getInt64(Offset));
    auto *Snapshot = Uniform.CreateAlignedLoad(L->getType(), Address, L->getAlign(), "cpu.constant");
    L->replaceAllUsesWith(Snapshot);
    L->eraseFromParent();
  }
  for (auto *R : returns) {
    BranchInst::Create(Latch, R);
    R->eraseFromParent();
  }
  B.SetInsertPoint(Latch);
  auto *Next = B.CreateAdd(Lane, B.getInt32(1), "cpu.lane.next");
  Lane->addIncoming(Next, Latch);
  B.CreateCondBr(B.CreateICmpULT(Next, B.getInt32(BlockSize)), Header, Exit);
  B.SetInsertPoint(Exit);
  // Match the runtime-visible coordinate state at ordinary CTA completion.
  B.CreateStore(B.getInt32(0), M.getNamedGlobal("intra_warp_index"));
  B.CreateStore(B.getInt32(0), M.getNamedGlobal("inter_warp_index"));
  B.CreateRetVoid();
  F.addFnAttr("cpu.region.schedule", "whole-lane-ssa");
}
}
