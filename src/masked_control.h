// Static, reducible-CFG predication before CuPBoP's phase-loop construction.
// Each source block has lane-private reachability; source loops become uniform
// loops that continue while any lane has a pending backedge. There is no lane
// program counter, interpreter, runtime lane thread, or workload dispatch.
#pragma once
#include "llvm/Analysis/LoopInfo.h"
#include <memory>

namespace cpu_masks {
struct Node {
  llvm::BasicBlock* Block;
  llvm::Loop* Loop;
  std::vector<Node> Children;
};

static std::vector<Node> schedule(llvm::Function& F, llvm::LoopInfo& LI,
                                llvm::Loop* Parent) {
  using namespace llvm;
  std::vector<Node> nodes;
  std::map<BasicBlock*,size_t> owner;
  std::map<Loop*,size_t> loops;
  for(auto& B:F) {
    if(Parent&&!Parent->contains(&B))continue;
    auto* L=LI.getLoopFor(&B);
    if(L!=Parent) {
      while(L&&L->getParentLoop()!=Parent)L=L->getParentLoop();
      if(!L)refuse("masked CFG has inconsistent loop nesting");
      if(!loops.count(L)) {
        loops[L]=nodes.size();nodes.push_back({L->getHeader(),L,{}});
      }
      owner[&B]=loops[L];
    } else {
      owner[&B]=nodes.size();nodes.push_back({&B,nullptr,{}});
    }
  }
  std::vector<std::set<size_t>> edges(nodes.size());
  std::vector<unsigned> incoming(nodes.size(),0);
  for(auto& B:F)if(owner.count(&B))for(auto* S:successors(&B)) {
    if(!owner.count(S)||(Parent&&S==Parent->getHeader()))continue;
    auto from=owner[&B],to=owner[S];if(from==to)continue;
    if(edges[from].insert(to).second)++incoming[to];
  }
  std::vector<Node> result;
  std::set<size_t> ready;
  for(size_t i=0;i<nodes.size();++i)if(!incoming[i])ready.insert(i);
  while(!ready.empty()) {
    auto i=*ready.begin();ready.erase(ready.begin());auto node=nodes[i];
    if(node.Loop)node.Children=schedule(F,LI,node.Loop);
    result.push_back(std::move(node));
    for(auto j:edges[i])if(!--incoming[j])ready.insert(j);
  }
  if(result.size()!=nodes.size())refuse("irreducible CFG requires supported reconvergence lowering");
  return result;
}

static bool phaseSynchronization(const llvm::Instruction& I) {
  auto* Call=llvm::dyn_cast<llvm::CallInst>(&I);
  auto* CF=Call?Call->getCalledFunction():nullptr;
  if(!CF)return false;
  auto N=CF->getName();
  return N=="llvm.nvvm.barrier0"||N=="llvm.nvvm.bar.warp.sync"||
         N=="llvm.nvvm.barrier.sync";
}

static bool run(llvm::Module& M,llvm::Function& F) {
  using namespace llvm;
  // The original CuPBoP algorithm assumes that a predicate between phase
  // loops is uniform. Both shared-memory ordering and source/generated
  // synchronization (including lowered shuffles) introduce phase boundaries.
  // Even a runtime-untaken collective arm can cut an earlier divergent EXIT
  // into phases: sampling lane zero's predicate then resurrects exited lanes.
  // Preserve per-lane reachability for every conditional CFG with such cuts.
  // This runs after lowerCollectives and before synthetic CuPBoP boundaries.
  bool phases=false,conditional=false;
  for(auto& I:instructions(F)) {
    phases|=bool(I.getMetadata("cpu.warp.memory.order"))||phaseSynchronization(I);
    if(auto* B=dyn_cast<BranchInst>(&I))conditional|=B->isConditional();
    conditional|=isa<SwitchInst>(I);
  }
  if(!phases||!conditional)return false;
  removeUnreachableBlocks(F);
  // Reachability changes at branches and joins, not at synchronization.
  // Coalesce unconditional chains before assigning pending masks, including
  // the artificial blocks introduced by split_block_by_sync. Every cut stays
  // in instruction order and flush() below still emits a separate phase.
  // Cross-cut value lifetimes are handled independently of control regions.
  unsigned coalesced=0;
  bool changed;
  do {
    changed=false;
    for(auto It=F.begin();It!=F.end();) {
      auto* B=&*It++;
      auto* P=B->getSinglePredecessor();
      if(!P||P==B)continue;
      auto* Br=dyn_cast<BranchInst>(P->getTerminator());
      if(!Br||!Br->isUnconditional())continue;
      if(MergeBlockIntoPredecessor(B)){++coalesced;changed=true;}
    }
  } while(changed);
  F.addFnAttr("cpu.mask.coalesced.blocks",std::to_string(coalesced));
  // Recompute analyses after coalescing; joins, backedges and synchronization
  // remain explicit and are handled by the same reducible-CFG scheduler.
  DominatorTree DT(F);LoopInfo LI(DT);
  auto plan=schedule(F,LI,nullptr);
  std::vector<BasicBlock*> original;
  for(auto& B:F)original.push_back(&B);
  // Moving source segments under guards invalidates SSA dominance across
  // both control regions and synchronization cuts within a region. Assign
  // segment identities before inserting spills: a unique reachability mask
  // does not make values live across a barrier uniform or lane-independent.
  std::map<Instruction*,unsigned> segments;
  unsigned segment=0;
  for(auto& B:F) {
    ++segment;
    for(auto& I:B) {
      segments[&I]=segment;
      if(phaseSynchronization(I))++segment;
    }
  }
  std::vector<Instruction*> crossing;
  for(auto& I:instructions(F)) {
    if(isa<AllocaInst>(&I)||I.isTerminator()||I.getType()->isVoidTy())continue;
    for(auto* U:I.users())if(auto* UI=dyn_cast<Instruction>(U))
      if(segments.at(UI)!=segments.at(&I)){crossing.push_back(&I);break;}
  }
  F.addFnAttr("cpu.mask.control.regions",std::to_string(original.size()));
  F.addFnAttr("cpu.mask.phase.spills",std::to_string(crossing.size()));
  for(auto* I:crossing)DemoteRegToStack(*I,false);
  auto& C=M.getContext();
  auto* Entry=BasicBlock::Create(C,"cpu.mask.entry",&F,&F.getEntryBlock());
  IRBuilder<> E(Entry);
  std::vector<AllocaInst*> allocas;
  for(auto* B:original)for(auto& I:*B)if(auto* A=dyn_cast<AllocaInst>(&I))allocas.push_back(A);
  for(auto* A:allocas) {
    if(!isa<ConstantInt>(A->getArraySize()))refuse("dynamic source stack allocation requires explicit lane storage");
    A->removeFromParent();A->insertInto(Entry,Entry->end());
  }
  std::map<BasicBlock*,AllocaInst*> masks;
  for(auto* B:original)masks[B]=E.CreateAlloca(E.getInt1Ty(),nullptr,B->getName()+".pending");
  auto* Running=E.CreateAlloca(E.getInt1Ty(),nullptr,"cpu.mask.running");
  for(auto* B:original)E.CreateStore(E.getInt1(B==original.front()),masks[B]);
  auto* Tail=Entry;
  auto barrier=[&](IRBuilder<>& B) {
    // Conservative CTA phases cover every original warp phase. This is a
    // shared-memory ordering choice, not a change in arithmetic/participants.
    B.CreateCall(M.getOrInsertFunction("llvm.nvvm.barrier0",B.getVoidTy()));
  };
  std::function<void(const std::vector<Node>&)> emit;
  emit=[&](const std::vector<Node>& nodes) {
    for(auto& N:nodes) {
      if(N.Loop) {
        auto* Test=BasicBlock::Create(C,"cpu.mask.loop.test",&F);
        auto* Body=BasicBlock::Create(C,"cpu.mask.loop.body",&F);
        auto* Exit=BasicBlock::Create(C,"cpu.mask.loop.exit",&F);
        IRBuilder<>(Tail).CreateBr(Test);IRBuilder<> T(Test);
        auto* Pending=T.CreateZExt(T.CreateLoad(T.getInt1Ty(),masks[N.Block]),T.getInt32Ty());
        auto* Any=T.CreateCall(M.getOrInsertFunction("barrier0_or",T.getInt32Ty(),T.getInt32Ty()),{Pending},"cpu.mask.any");
        T.CreateCondBr(T.CreateICmpNE(Any,T.getInt32(0)),Body,Exit);
        Tail=Body;emit(N.Children);IRBuilder<>(Tail).CreateBr(Test);Tail=Exit;
        continue;
      }
      auto* Old=N.Block;
      IRBuilder<> S(Tail);
      S.CreateStore(S.CreateLoad(S.getInt1Ty(),masks[Old]),Running);
      S.CreateStore(S.getFalse(),masks[Old]);
      std::vector<Instruction*> segment;
      auto flush=[&] {
        if(segment.empty())return;
        auto* Body=BasicBlock::Create(C,Old->getName()+".masked",&F);
        auto* Next=BasicBlock::Create(C,"cpu.mask.continue",&F);
        IRBuilder<> G(Tail);auto* Guard=G.CreateCondBr(G.CreateLoad(G.getInt1Ty(),Running),Body,Next);
        Guard->setMetadata("cpu.lane.guard",MDNode::get(C,{}));
        for(auto* I:segment){I->removeFromParent();I->insertInto(Body,Body->end());}
        IRBuilder<>(Body).CreateBr(Next);segment.clear();Tail=Next;
      };
      // Emit outgoing per-lane reachability in the same guarded segment as
      // its predicate; do not speculate loads/calculation on inactive lanes.
      auto* Term=Old->getTerminator();IRBuilder<> EB(Term);
      if(auto* Br=dyn_cast<BranchInst>(Term)) {
        if(Br->isUnconditional()||Br->getSuccessor(0)==Br->getSuccessor(1))EB.CreateStore(EB.getTrue(),masks[Br->getSuccessor(0)]);
        else {
          EB.CreateStore(Br->getCondition(),masks[Br->getSuccessor(0)]);
          EB.CreateStore(EB.CreateNot(Br->getCondition()),masks[Br->getSuccessor(1)]);
        }
      } else if(auto* Switch=dyn_cast<SwitchInst>(Term)) {
        std::map<BasicBlock*,Value*> conditions;
        Value* Any=EB.getFalse();
        for(auto Case:Switch->cases()) {
          auto* Match=EB.CreateICmpEQ(Switch->getCondition(),Case.getCaseValue());
          Any=EB.CreateOr(Any,Match);auto* Dest=Case.getCaseSuccessor();
          conditions[Dest]=conditions.count(Dest)?EB.CreateOr(conditions[Dest],Match):Match;
        }
        auto* Default=Switch->getDefaultDest();auto* Otherwise=EB.CreateNot(Any);
        conditions[Default]=conditions.count(Default)?EB.CreateOr(conditions[Default],Otherwise):Otherwise;
        // Function order makes output deterministic, including duplicate case
        // destinations and a default destination shared with explicit cases.
        for(auto* Dest:original)if(conditions.count(Dest))EB.CreateStore(conditions[Dest],masks[Dest]);
      } else if(!isa<ReturnInst>(Term))refuse("masked CFG requires branch/switch/return terminators");
      for(auto It=Old->begin();It!=Old->end();) {
        auto* I=&*It++;if(I->isTerminator())break;
        if(phaseSynchronization(*I)) {
          flush();IRBuilder<> B(Tail);barrier(B);I->eraseFromParent();continue;
        }
        segment.push_back(I);
      }
      flush();
    }
  };
  emit(plan);IRBuilder<>(Tail).CreateRetVoid();
  for(auto* B:original)B->dropAllReferences();
  for(auto* B:original)B->eraseFromParent();
  F.addFnAttr("cpu.masked.control","static-reducible-cta");
  if(verifyFunction(F,&errs()))refuse("masked control-flow verification failed");
  return true;
}
} // namespace cpu_masks
