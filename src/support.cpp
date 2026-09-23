// CPU-independent support for the imported CuPBoP region/loop passes.
#include "tool.h"
#include "cg_sync.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Local.h"
#include <set>
#include <vector>
using namespace llvm;
bool isKernelFunction(Module*, Function* F) { return F->hasFnAttribute("cpu.coarsen.kernel"); }
void VerifyModule(Module* M) { if(verifyModule(*M,&errs())) report_fatal_error("invalid coarsening IR"); }
void printIR(Module* M) { if(cupbop_debug()) M->print(errs(),nullptr); }
LoadInst* createLoad(IRBuilder<>& B, Value* P, bool V) {
  Type* T=nullptr;
  if(auto* A=dyn_cast<AllocaInst>(P)) T=A->getAllocatedType();
  else if(auto* G=dyn_cast<GlobalVariable>(P)) T=G->getValueType();
  else if(auto* G=dyn_cast<GetElementPtrInst>(P)) T=G->getResultElementType();
  if(!T) report_fatal_error("coarsening: unknown load element type");
  return B.CreateLoad(T,P,V);
}
static Type* element(Value* P) {
  if(auto* A=dyn_cast<AllocaInst>(P))return A->getAllocatedType();
  if(auto* G=dyn_cast<GlobalVariable>(P))return G->getValueType();
  if(auto* G=dyn_cast<GetElementPtrInst>(P))return G->getResultElementType();
  report_fatal_error("coarsening: unknown GEP element type");
}
Value* createGEP(IRBuilder<>& B, Value* P, ArrayRef<Value*> I) {return B.CreateGEP(element(P),P,I);}
Value* createInBoundsGEP(IRBuilder<>& B, Value* P, ArrayRef<Value*> I) {return B.CreateInBoundsGEP(element(P),P,I);}
void replace_block(Function* F,BasicBlock* A,BasicBlock* Z) {
  for(auto& B:*F)B.getTerminator()->replaceUsesOfWith(A,Z);
}
CallInst* CreateInterWarpBarrier(Instruction* I) {
  IRBuilder<> B(I);auto* C=B.CreateCall(I->getModule()->getOrInsertFunction("llvm.nvvm.barrier0",B.getVoidTy()));
  C->setMetadata("cpu.synthetic.boundary",MDNode::get(I->getContext(),{}));return C;
}
CallInst* CreateIntraWarpBarrier(Instruction* I) {
  IRBuilder<> B(I);auto* C=B.CreateCall(I->getModule()->getOrInsertFunction("llvm.nvvm.bar.warp.sync",B.getVoidTy(),B.getInt32Ty()),{B.getInt32(-1)});
  C->setMetadata("cpu.synthetic.boundary",MDNode::get(I->getContext(),{}));return C;
}
static bool barrier(BasicBlock* B,bool warp,bool block) {
  for(auto& I:*B)if(auto* C=dyn_cast<CallInst>(&I))if(auto* F=C->getCalledFunction()) {
    auto S=F->getName();
    if(warp&&(isWarpSync(S.str())||S=="cupbop.shfl.barrier"))return true;
    if(block&&(S=="llvm.nvvm.barrier0"||S=="llvm.nvvm.barrier.sync"||isCGSync(S.str())))return true;
  }
  return false;
}
bool has_warp_barrier(BasicBlock* B){return barrier(B,true,false);}
bool has_block_barrier(BasicBlock* B){return barrier(B,false,true);}
bool has_barrier(BasicBlock* B){return barrier(B,true,true);}
bool has_barrier(Function* F){for(auto& B:*F)if(has_barrier(&B))return true;return false;}
static bool findBarrier(BasicBlock* A,BasicBlock* Z,bool blockOnly) {
  std::vector<BasicBlock*> pending;for(auto* S:successors(A))pending.push_back(S);
  std::set<BasicBlock*> seen;
  while(!pending.empty()) {auto* B=pending.back();pending.pop_back();
    if(B==Z||!seen.insert(B).second)continue;
    if(blockOnly?has_block_barrier(B):has_barrier(B))return true;
    for(auto* S:successors(B))pending.push_back(S);
  }return false;
}
bool find_block_barrier_in_region(BasicBlock* A,BasicBlock* Z){return findBarrier(A,Z,true);}
bool find_barrier_in_region(BasicBlock* A,BasicBlock* Z){return findBarrier(A,Z,false);}
