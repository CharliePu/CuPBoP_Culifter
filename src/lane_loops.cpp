#include "lane_loops.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <optional>

using namespace llvm;
namespace cpu_schedule {
namespace {
bool directOnly(GlobalVariable &G) {
  for (auto *U : G.users()) {
    if (auto *L = dyn_cast<LoadInst>(U)) {
      if (L->getPointerOperand() != &G || !L->isSimple())
        return false;
    } else if (auto *S = dyn_cast<StoreInst>(U)) {
      if (S->getPointerOperand() != &G || S->getValueOperand() == &G || !S->isSimple())
        return false;
    } else {
      return false; // address escapes (GEP, cast, call argument, stored value)
    }
  }
  return true;
}

// Runtime-owned read-only launch state and the parameter bank. Kernel data
// never lives in these objects; lane loops only read them.
bool runtimeState(const GlobalVariable &G) {
  for (auto *Name : {"block_size", "block_size_x", "block_size_y", "block_size_z",
                     "grid_size_x", "grid_size_y", "grid_size_z", "block_index_x",
                     "block_index_y", "block_index_z", "cpu_shared_memory", "const_mem"})
    if (G.getName() == Name)
      return true;
  return false;
}
} // namespace

LaneIndices localizeLaneIndices(Function &F) {
  auto &M = *F.getParent();
  LaneIndices Result;
  // Calls with non-fallthrough control flow have no single publication point.
  for (auto &I : instructions(F))
    if (isa<InvokeInst>(I) || isa<CallBrInst>(I))
      return Result;
  for (StringRef Name : {"intra_warp_index", "inter_warp_index"}) {
    auto *G = M.getNamedGlobal(Name);
    if (!G || G->user_empty() || !directOnly(*G))
      continue;
    bool Local = true;
    for (auto *U : G->users())
      Local &= cast<Instruction>(U)->getFunction() == &F;
    if (!Local)
      continue; // another function reads or writes the TLS index directly
    SmallVector<Instruction *, 64> Accesses;
    for (auto *U : G->users())
      Accesses.push_back(cast<Instruction>(U));
    auto *Type = G->getValueType();
    IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
    auto *Slot = B.CreateAlloca(Type, nullptr, Name + ".local");
    B.CreateStore(B.CreateLoad(Type, G, Name + ".entry"), Slot);
    for (auto *I : Accesses) {
      if (auto *L = dyn_cast<LoadInst>(I))
        L->setOperand(L->getPointerOperandIndex(), Slot);
      else
        cast<StoreInst>(I)->setOperand(StoreInst::getPointerOperandIndex(), Slot);
    }
    // Publish the current index wherever code outside this function could
    // observe the TLS object, and adopt any value such code leaves behind.
    SmallVector<Instruction *, 16> Boundaries;
    for (auto &I : instructions(F))
      if ((isa<CallBase>(I) && !isa<IntrinsicInst>(I)) || isa<ReturnInst>(I))
        Boundaries.push_back(&I);
    for (auto *I : Boundaries) {
      IRBuilder<> Before(I);
      Before.CreateStore(Before.CreateLoad(Type, Slot), G);
      if (isa<ReturnInst>(I))
        continue;
      IRBuilder<> After(I->getNextNode());
      After.CreateStore(After.CreateLoad(Type, G), Slot);
    }
    (Name == "intra_warp_index" ? Result.intra : Result.inter) = Slot;
  }
  if (!Result.count())
    return Result;
  // Every value stored to an index is in [0, 1024]: schedule 0 starts lane and
  // warp loops at 0, steps by 1 and exits at block_size <= 1024 (lanes) or
  // ceil(block_size / 32) <= 32 (warps); the runtime enters with both at 0.
  // Coordinate arithmetic built only from index loads and non-negative
  // constants therefore cannot wrap when its bound fits the type. Recording
  // nuw/nsw lets scalar evolution see affine lane addresses after the
  // implicit sign extension of a 32-bit lane-array index.
  DenseMap<Value *, std::optional<uint64_t>> Bounds;
  std::function<std::optional<uint64_t>(Value *, unsigned)> Bound =
      [&](Value *V, unsigned Depth) -> std::optional<uint64_t> {
    if (auto *C = dyn_cast<ConstantInt>(V))
      return C->isNegative() ? std::nullopt : std::optional<uint64_t>(C->getZExtValue());
    auto It = Bounds.find(V);
    if (It != Bounds.end())
      return It->second;
    std::optional<uint64_t> R;
    if (auto *L = dyn_cast<LoadInst>(V)) {
      if (L->getPointerOperand() == Result.intra || L->getPointerOperand() == Result.inter)
        R = 1024;
    } else if (auto *O = dyn_cast<BinaryOperator>(V); O && Depth < 8) {
      auto A = Bound(O->getOperand(0), Depth + 1), B = Bound(O->getOperand(1), Depth + 1);
      if (A && B && *A < (1ull << 31) && *B < (1ull << 31)) {
        if (O->getOpcode() == Instruction::Add) R = *A + *B;
        if (O->getOpcode() == Instruction::Mul) R = *A * *B;
        if (O->getOpcode() == Instruction::Shl && *B < 31) R = *A << *B;
      }
    }
    return Bounds[V] = R;
  };
  for (auto &I : instructions(F))
    if (auto *O = dyn_cast<OverflowingBinaryOperator>(&I))
      if (O->getType()->isIntegerTy() && O->getType()->getIntegerBitWidth() >= 32)
        if (auto B = Bound(O, 0); B && *B < (1ull << 31)) {
          cast<BinaryOperator>(O)->setHasNoUnsignedWrap(true);
          cast<BinaryOperator>(O)->setHasNoSignedWrap(true);
        }
  return Result;
}

namespace {
bool slotLoad(Value *V, AllocaInst *Slot) {
  auto *L = dyn_cast<LoadInst>(V);
  return Slot && L && L->getPointerOperand() == Slot;
}

// The logical thread index intra + 32 * inter, as emitted by CuPBoP's context
// save/restore and by the adapter's coordinate helper. Lane loops never run
// intra past the warp width when a warp loop exists, and the warp index is
// fixed while the lane loop runs, so distinct iterations of either loop see
// distinct thread indices.
bool laneIndex(Value *V, const LaneIndices &X) {
  auto *A = dyn_cast<BinaryOperator>(V);
  if (!A || A->getOpcode() != Instruction::Add)
    return false;
  auto warp = [&](Value *W) {
    auto *O = dyn_cast<BinaryOperator>(W);
    if (!O)
      return false;
    for (unsigned i = 0; i != 2; ++i) {
      auto *C = dyn_cast<ConstantInt>(O->getOperand(1 - i));
      if (!C || !slotLoad(O->getOperand(i), X.inter))
        continue;
      if (O->getOpcode() == Instruction::Mul && C->equalsInt(32))
        return true;
      if (O->getOpcode() == Instruction::Shl && i == 0 && C->equalsInt(5))
        return true;
    }
    return false;
  };
  return (slotLoad(A->getOperand(0), X.intra) && warp(A->getOperand(1))) ||
         (slotLoad(A->getOperand(1), X.intra) && warp(A->getOperand(0)));
}

// A per-lane array: an alloca reached only through direct accesses or
// single-index GEPs whose results are only access addresses.
bool privateArray(AllocaInst &A) {
  auto address = [&](User *U, Value *P) {
    if (auto *L = dyn_cast<LoadInst>(U))
      return L->getPointerOperand() == P;
    if (auto *S = dyn_cast<StoreInst>(U))
      return S->getPointerOperand() == P && S->getValueOperand() != P;
    auto *I = dyn_cast<Instruction>(U);
    return I && I->isLifetimeStartOrEnd();
  };
  for (auto *U : A.users()) {
    if (auto *G = dyn_cast<GetElementPtrInst>(U)) {
      if (G->getPointerOperand() != &A)
        return false;
      for (auto *GU : G->users())
        if (!address(GU, G))
          return false;
    } else if (!address(U, &A)) {
      return false;
    }
  }
  return true;
}

// P addresses element [thread index] of a per-lane array and the access fits
// that element.
bool ownLaneAccess(Value *P, Type *Accessed, const LaneIndices &X, const DataLayout &DL) {
  auto *G = dyn_cast<GetElementPtrInst>(P);
  if (!G || G->getNumIndices() != 1)
    return false;
  auto *A = dyn_cast<AllocaInst>(G->getPointerOperand());
  if (!A || A == X.intra || A == X.inter || G->getSourceElementType() != A->getAllocatedType())
    return false;
  if (DL.getTypeStoreSize(Accessed).getFixedValue() >
      DL.getTypeAllocSize(A->getAllocatedType()).getFixedValue())
    return false;
  return laneIndex(G->getOperand(1), X) && privateArray(*A);
}

// Rebuild a lane-loop ID: optionally drop the parallel-access proposal and
// optionally request vectorization. insert_warp_loop's llvm.loop.disable_nonforced
// (the infinite-loop protection) is always kept, so without the request the
// loop vectorizer leaves the lane loop scalar.
MDNode *rebuildLoopID(MDNode *ID, bool KeepParallel, bool ForceVectorize) {
  auto &C = ID->getContext();
  SmallVector<Metadata *, 4> Ops{nullptr};
  for (unsigned i = 1; i < ID->getNumOperands(); ++i) {
    auto *N = dyn_cast<MDNode>(ID->getOperand(i));
    auto *S = N && N->getNumOperands() ? dyn_cast<MDString>(N->getOperand(0)) : nullptr;
    if (S && S->getString() == "llvm.loop.parallel_accesses" && !KeepParallel)
      continue;
    Ops.push_back(ID->getOperand(i));
  }
  if (ForceVectorize)
    Ops.push_back(MDNode::get(C, {MDString::get(C, "llvm.loop.vectorize.enable"),
                                  ConstantAsMetadata::get(ConstantInt::getTrue(C))}));
  auto *New = MDNode::getDistinct(C, Ops);
  New->replaceOperandWith(0, New);
  return New;
}

MDNode *proposedGroup(MDNode *ID) {
  for (unsigned i = 1; ID && i < ID->getNumOperands(); ++i)
    if (auto *N = dyn_cast<MDNode>(ID->getOperand(i)))
      if (N->getNumOperands() == 2)
        if (auto *S = dyn_cast<MDString>(N->getOperand(0)))
          if (S->getString() == "llvm.loop.parallel_accesses")
            return dyn_cast<MDNode>(N->getOperand(1));
  return nullptr;
}
} // namespace

unsigned widenLaneMasks(Function &F) {
  SmallVector<AllocaInst *, 16> Masks;
  for (auto &I : F.getEntryBlock())
    if (auto *A = dyn_cast<AllocaInst>(&I))
      if (A->getAllocatedType()->isIntegerTy(1) && privateArray(*A))
        Masks.push_back(A);
  auto *Byte = Type::getInt8Ty(F.getContext());
  for (auto *A : Masks) {
    IRBuilder<> B(A);
    auto *Wide = B.CreateAlloca(Byte, A->getArraySize(), A->getName() + ".byte");
    Wide->setAlignment(A->getAlign());
    // Rewrite one access through pointer P (A itself or a GEP of it) to Q.
    auto rewrite = [&](Instruction *U, Value *Q) {
      IRBuilder<> At(U);
      if (auto *L = dyn_cast<LoadInst>(U)) {
        auto *N = At.CreateLoad(Byte, Q);
        N->copyMetadata(*L);
        N->setAlignment(L->getAlign());
        L->replaceAllUsesWith(At.CreateTrunc(N, L->getType()));
      } else if (auto *S = dyn_cast<StoreInst>(U)) {
        auto *N = At.CreateStore(At.CreateZExt(S->getValueOperand(), Byte), Q);
        N->copyMetadata(*S);
        N->setAlignment(S->getAlign());
      } else {
        auto *N = U->clone();
        N->insertBefore(U);
        N->replaceUsesOfWith(U->getOperand(1), Q); // lifetime marker pointer
      }
      U->eraseFromParent();
    };
    for (auto *U : SmallVector<User *, 16>(A->users())) {
      if (auto *G = dyn_cast<GetElementPtrInst>(U)) {
        IRBuilder<> At(G);
        SmallVector<Value *, 1> Index(G->indices());
        auto *H = At.CreateGEP(Byte, Wide, Index, G->getName() + ".byte");
        for (auto *GU : SmallVector<User *, 8>(G->users()))
          rewrite(cast<Instruction>(GU), H);
        G->eraseFromParent();
      } else {
        rewrite(cast<Instruction>(U), Wide);
      }
    }
    A->eraseFromParent();
  }
  return Masks.size();
}

ParallelLaneLoops annotateParallelLaneLoops(Function &F, const LaneIndices &X, bool Prove,
                                            bool ForceVectorize) {
  ParallelLaneLoops Summary;
  const auto &DL = F.getParent()->getDataLayout();
  DominatorTree DT(F);
  LoopInfo LI(DT);
  struct Decision {
    Instruction *Branch;
    MDNode *Group;
    std::string Refusal;
    SmallVector<Instruction *, 32> Tagged;
    bool SubByte = false;
  };
  std::vector<Decision> Decisions;
  for (auto *L : LI.getLoopsInPreorder()) {
    // insert_warp_loop attaches the proposal to the lane-loop condition.
    auto *Branch = L->getHeader()->getTerminator();
    auto *Group = proposedGroup(Branch->getMetadata(LLVMContext::MD_loop));
    if (!Group)
      continue;
    ++Summary.laneLoops;
    Decision D{Branch, Group, "", {}};
    unsigned KernelLoads = 0, KernelStores = 0;
    auto refuse = [&](const Twine &Why) {
      if (D.Refusal.empty())
        D.Refusal = Why.str();
    };
    if (!Prove)
      refuse("parallel assertion disabled (--parallel-lanes=false)");
    if (!X.intra || !X.inter)
      refuse("lane indices are not function-local");
    if (!L->isInnermost())
      refuse("contains a kernel loop (a lane may execute a store repeatedly)");
    for (auto *BB : L->blocks())
      for (auto &I : *BB) {
        if (!I.mayReadOrWriteMemory())
          continue;
        if (auto *Call = dyn_cast<CallBase>(&I)) {
          if (I.isLifetimeStartOrEnd() &&
              ownLaneAccess(Call->getArgOperand(1), Type::getInt8Ty(F.getContext()), X, DL)) {
            D.Tagged.push_back(&I);
            continue;
          }
          auto *Callee = Call->getCalledFunction();
          refuse("call to " + (Callee ? Callee->getName() : StringRef("an indirect target")) +
                 " (collective, runtime helper or unmodeled effects)");
          continue;
        }
        Value *P = nullptr;
        Type *T = nullptr;
        bool Store = false;
        if (auto *Ld = dyn_cast<LoadInst>(&I)) {
          if (!Ld->isSimple()) refuse("atomic or volatile load");
          P = Ld->getPointerOperand(), T = Ld->getType();
        } else if (auto *St = dyn_cast<StoreInst>(&I)) {
          if (!St->isSimple()) refuse("atomic or volatile store");
          P = St->getPointerOperand(), T = St->getValueOperand()->getType(), Store = true;
        } else {
          refuse(Twine(I.getOpcodeName()) + " (atomic read-modify-write or fence)");
          continue;
        }
        // LLVM's vectorizer scalarizes every access of an irregular type
        // (store size != type size, e.g. i1 lane masks) into predicated
        // per-lane code; such loops are never forced.
        D.SubByte |= !DL.typeSizeEqualsStoreSize(T);
        if (P == X.intra || P == X.inter)
          continue; // the loop's own induction state; promoted to SSA by LLVM
        auto *Object = getUnderlyingObject(P, 0);
        if (isa<AllocaInst>(Object)) {
          if (!ownLaneAccess(P, T, X, DL))
            refuse("function-local storage not addressed by the lane's own thread index "
                   "(cross-lane exchange or shared flag)");
          D.Tagged.push_back(&I);
          continue;
        }
        if (auto *G = dyn_cast<GlobalVariable>(Object); G && runtimeState(*G)) {
          if (Store) refuse("store to runtime launch state");
          D.Tagged.push_back(&I);
          continue;
        }
        (Store ? KernelStores : KernelLoads) += 1;
        D.Tagged.push_back(&I);
      }
    // Global, shared and local (kernel) memory may alias arbitrarily, including
    // through generic shared-memory pointers. Without addresses, allow only
    // patterns in which no lane can observe or overwrite another lane's
    // kernel-memory effect differently under a lane-interleaved order:
    // read-only regions, or a single store instruction per lane with no
    // kernel-memory reads (same-address writes stay in lane order).
    if (KernelStores && KernelLoads)
      refuse("reads and writes kernel memory (a lane could read another lane's write)");
    else if (KernelStores > 1)
      refuse("more than one kernel-memory store (cross-lane write order not provable)");
    errs() << "{\"lane_loop\":\"" << L->getHeader()->getName() << "\",\"kernel_loads\":" << KernelLoads
           << ",\"kernel_stores\":" << KernelStores << ",\"parallel\":" << (D.Refusal.empty() ? "true" : "false")
           << ",\"sub_byte_lane_storage\":" << (D.SubByte ? "true" : "false")
           << ",\"refusal\":\"" << D.Refusal << "\"}\n";
    Decisions.push_back(std::move(D));
  }
  SmallPtrSet<Instruction *, 32> Asserted;
  for (auto &D : Decisions) {
    if (!D.Refusal.empty())
      continue;
    ++Summary.asserted;
    Asserted.insert(D.Branch);
    // Proved parallel. Vectorization is requested only on demand: forced
    // vectorization bypasses LLVM's cost model (disable_nonforced leaves no
    // cost-model-driven mode), and it was measured slower on pooling and
    // softmax kernels. Never force loops with sub-byte lane storage.
    bool Force = ForceVectorize && !D.SubByte;
    D.Branch->setMetadata(LLVMContext::MD_loop,
                          rebuildLoopID(D.Branch->getMetadata(LLVMContext::MD_loop), true, Force));
    Summary.forced += Force;
    for (auto *I : D.Tagged)
      I->setMetadata(LLVMContext::MD_access_group,
                     uniteAccessGroups(I->getMetadata(LLVMContext::MD_access_group), D.Group));
  }
  // Remove every proposal that was not proved, including any whose branch is
  // not a loop header in the final CFG.
  for (auto &I : instructions(F))
    if (auto *ID = I.getMetadata(LLVMContext::MD_loop))
      if (proposedGroup(ID) && !Asserted.count(&I))
        I.setMetadata(LLVMContext::MD_loop, rebuildLoopID(ID, false, false));
  return Summary;
}
} // namespace cpu_schedule
