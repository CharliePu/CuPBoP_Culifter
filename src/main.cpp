// CuLifter ABI adapter for CuPBoP_Vortex's hierarchical collapsing passes.
#include "tool.h"
#include "insert_sync.h"
#include "handle_sync.h"
#include "insert_warp_loop.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include <map>
#include <set>
#include <stdexcept>
using namespace llvm;
static cl::opt<std::string> Input(cl::Positional,cl::Required);
static cl::opt<std::string> Output("o",cl::Required);
static cl::opt<std::string> Kernel("kernel",cl::Required);
static cl::opt<std::string> Rename("rename",cl::init(""));
static cl::opt<std::string> Target("target",cl::init("x86_64-linux-gnu"));
static cl::opt<std::string> Dump("dump-prefix",cl::init(""));
static cl::opt<unsigned> BlockSize("block-size",cl::init(32));
static cl::opt<unsigned> SharedBytes("shared-memory-bytes",cl::init(32768));
static void refuse(const Twine& S){throw std::runtime_error(S.str());}
static GlobalVariable* global(Module& M,StringRef N,Type* T) {
  if(auto* G=M.getNamedGlobal(N))return G;
  return new GlobalVariable(M,T,false,GlobalValue::ExternalLinkage,nullptr,N,nullptr,GlobalVariable::GeneralDynamicTLSModel);
}
static Value* get(IRBuilder<>& B,Module& M,StringRef N){auto* G=M.getNamedGlobal(N);return B.CreateLoad(G->getValueType(),G,N+".value");}
static Value* tid(IRBuilder<>& B,Module& M){return B.CreateAdd(get(B,M,"intra_warp_index"),B.CreateMul(get(B,M,"inter_warp_index"),B.getInt32(32)),"cpu.thread");}
// Logical coordinates must be independent of the chosen loop schedule. A
// flat CTA loop uses intra_warp_index across the whole CTA; a nested schedule
// uses inter*32+intra. Both represent the same logical lane and warp base.
static Value* lane(IRBuilder<>& B,Module& M){return B.CreateAnd(tid(B,M),B.getInt32(31));}
static Value* warpBase(IRBuilder<>& B,Module& M){return B.CreateAnd(tid(B,M),B.getInt32(~31u));}
static void dump(Module& M,StringRef S){if(Dump.empty())return;std::error_code E;raw_fd_ostream O(Dump+S.str()+".ll",E);M.print(O,nullptr);}
static uint64_t constant(Value* V,const char* What){if(auto* C=dyn_cast<ConstantInt>(V))return C->getZExtValue();refuse(Twine("dynamic ")+What);return 0;}
static void marker(IRBuilder<>& B,Module& M){B.CreateCall(M.getOrInsertFunction("llvm.nvvm.bar.warp.sync",B.getVoidTy(),B.getInt32Ty()),{B.getInt32(-1)});}
#include "fp16_collective.h"
#include "masked_control.h"
static bool shuffleName(StringRef N){return N=="shfl_bfly_f32"||N=="shfl_bfly_i32"||N=="shfl_down_f32"||N=="shfl_down_i32"||N=="shfl_up_f32"||N=="shfl_up_i32"||N=="shfl_idx_f32"||N=="shfl_idx_i32"||N.starts_with("llvm.nvvm.shfl.sync.");}
static bool tensorName(StringRef N){return N=="ldsm_m88_warp"||N=="hmma1688_tf32_warp"||N=="hmma16816_f16_warp";}
static void preserveSharedInstructionOrder(Module& M,Function& F){
  bool requested=false;std::vector<CallInst*> hints;
  for(auto& I:instructions(F))if(auto* C=dyn_cast<CallInst>(&I))if(auto* CF=C->getCalledFunction())
    if(CF->getName()=="shared_turn_begin"||CF->getName()=="shared_turn_end"){requested=true;hints.push_back(C);}
  if(!requested)return;
  for(auto* C:hints)C->eraseFromParent();
  auto* Shared=M.getNamedGlobal("shared_mem");if(!Shared)return;
  // SASS orders shared-memory instructions within a converged warp. Preserve
  // RAW/WAR/WAW ordering between shared accesses. Distinct writes may alias
  // across virtual lanes even when ordinary single-lane alias analysis says
  // otherwise. Read-only groups need no cut. Explicit collectives cut phases.
  std::map<BasicBlock*,unsigned> out;SmallPtrSet<Instruction*,32> cuts;
  bool changed=true;unsigned rounds=0;
  while(changed){
    if(++rounds>10000)refuse("shared-memory phase analysis failed to converge");changed=false;
    for(auto& BB:F){unsigned state=0;for(auto* P:predecessors(&BB))state|=out[P];
      for(auto& I:BB){
        if(auto* C=dyn_cast<CallInst>(&I))if(auto* CF=C->getCalledFunction()){
          auto N=CF->getName();if(N=="syncthreads"||N.starts_with("llvm.nvvm.bar")||tensorName(N)||shuffleName(N)||N.starts_with("barrier0_"))state=0;
        }
        Value* P=nullptr;unsigned kind=0;
        if(auto* L=dyn_cast<LoadInst>(&I)){P=L->getPointerOperand();kind=1;}
        if(auto* S=dyn_cast<StoreInst>(&I)){P=S->getPointerOperand();kind=2;}
        if(!P||getUnderlyingObject(P)!=Shared)continue;
        if(cuts.count(&I))state=0;
        if((kind==1&&(state&2))||(kind==2&&state)){
          if(cuts.insert(&I).second)changed=true;state=0;
        }
        state|=kind;
      }
      if(out[&BB]!=state){out[&BB]=state;changed=true;}
    }
  }
  // Source order ensures deterministic output. These boundaries cover only
  // launched lanes; the final partial warp is bounded by the launch volume.
  std::vector<Instruction*> ordered;for(auto& I:instructions(F))if(cuts.count(&I))ordered.push_back(&I);
  for(auto* I:ordered){IRBuilder<> B(I);auto* C=B.CreateCall(M.getOrInsertFunction("llvm.nvvm.bar.warp.sync",B.getVoidTy(),B.getInt32Ty()),{B.getInt32(-1)});
    C->setMetadata("cpu.warp.memory.order",MDNode::get(M.getContext(),{}));}
}
static void normalizeHelpers(Module& M,Function& F){
  // CuLifter emits local helper bodies for scalar packing around collectives.
  // Expose the collective in the entry CFG before region discovery; never
  // inline unsupported-instruction placeholders into apparently valid no-ops.
  bool changed=true; unsigned inlined=0;
  while(changed){changed=false;std::vector<CallBase*> calls;
    for(auto& I:instructions(F))if(auto* C=dyn_cast<CallBase>(&I))calls.push_back(C);
    for(auto* C:calls){auto* CF=C->getCalledFunction();if(!CF)continue;
      if(CF->getName().starts_with("sass_unsupported_"))refuse(Twine("unsupported lifted instruction: ")+CF->getName());
      if(!CF->isDeclaration()){
        if(CF==&F||++inlined>10000)refuse("recursive or excessive helper expansion");
        InlineFunctionInfo Info;if(!InlineFunction(*C,Info).isSuccess())refuse("helper inlining failed");changed=true;
      }
    }
  }
  preserveSharedInstructionOrder(M,F);
  std::vector<CallInst*> syncs;
  for(auto& I:instructions(F))if(auto* C=dyn_cast<CallInst>(&I))if(auto* CF=C->getCalledFunction())
    if(CF->getName()=="syncthreads"||CF->getName()=="warp_turn_begin"||CF->getName()=="warp_turn_end")syncs.push_back(C);
  for(auto* C:syncs){IRBuilder<> B(C);
    if(C->getCalledFunction()->getName()=="syncthreads")B.CreateCall(M.getOrInsertFunction("llvm.nvvm.barrier0",B.getVoidTy()));
    else {
      auto* Order=B.CreateCall(M.getOrInsertFunction("llvm.nvvm.bar.warp.sync",B.getVoidTy(),B.getInt32Ty()),{B.getInt32(-1)});
      Order->setMetadata("cpu.warp.memory.order",MDNode::get(M.getContext(),{}));
    }
    C->eraseFromParent();}
}
static void simplifyScalarIR(Function& F,bool Arithmetic=true){
  for(auto& I:instructions(F))if(auto* C=dyn_cast<CallInst>(&I))if(auto* CF=C->getCalledFunction()){
    auto N=CF->getName();if(shuffleName(N)||tensorName(N)||N.starts_with("barrier0_")||N.starts_with("llvm.nvvm.bar"))C->setConvergent();
  }
  LoopAnalysisManager LAM;FunctionAnalysisManager FAM;CGSCCAnalysisManager CGAM;ModuleAnalysisManager MAM;
  PassBuilder PB;PB.registerModuleAnalyses(MAM);PB.registerCGSCCAnalyses(CGAM);PB.registerFunctionAnalyses(FAM);PB.registerLoopAnalyses(LAM);PB.crossRegisterProxies(LAM,FAM,CGAM,MAM);
  FunctionPassManager PM;
  if(Arithmetic)PM.addPass(InstCombinePass(InstCombineOptions().setVerifyFixpoint(false).setMaxIterations(8)));
  // CuPBoP's loop-boundary pass requires a unique preheader and latch. Its
  // legacy pass declaration does not establish that prerequisite itself.
  // Multi-entry cycles need a common control header before the static lane
  // schedule can use natural loops. LLVM performs semantics-preserving CFG
  // restructuring; source PHIs are subsequently demoted per virtual lane.
  if(auto E=PB.parsePassPipeline(PM,"fix-irreducible,loop-simplify,lcssa,dce"))refuse(toString(std::move(E)));
  PM.run(F,FAM);
}
static void prepareGlobals(Module& M){
  auto* I=Type::getInt32Ty(M.getContext());
  for(auto N:{"block_size","block_size_x","block_size_y","block_size_z","grid_size_x","grid_size_y","grid_size_z",
              "block_index_x","block_index_y","block_index_z","intra_warp_index","inter_warp_index"})global(M,N,I);
  if(auto* G=M.getNamedGlobal("shared_mem"))G->setThreadLocalMode(GlobalVariable::GeneralDynamicTLSModel);
}
static void normalizeRegisters(Module& M,Function& F){
  std::vector<LoadInst*> loads;for(auto& I:instructions(F))if(auto* L=dyn_cast<LoadInst>(&I))loads.push_back(L);
  std::map<int,StringRef> uniform{{0,"block_size_x"},{4,"block_size_y"},{8,"block_size_z"},
      {12,"grid_size_x"},{16,"grid_size_y"},{20,"grid_size_z"},{32,"block_index_x"},{36,"block_index_y"},{40,"block_index_z"}};
  for(auto* L:loads){
    int64_t offset=0;auto* P=GetPointerBaseWithConstantOffset(L->getPointerOperand(),offset,M.getDataLayout());
    auto* G=dyn_cast<GlobalVariable>(P);if(!G||G->getName()!="const_mem")continue;
    if(offset>=0x100)continue;
    // Modern CUDA's generic shared pointer is c[0][0x18:0x20]. The
    // compile-only shared_address probe establishes this for SM75/SM89.
    // Map the whole pointer and either limb to the same block-local storage
    // used by LDS/STS; retaining the captured GPU pointer is never valid.
    if(offset==24||offset==28) {
      if((offset==28&&!L->getType()->isIntegerTy(32))||
         (!L->getType()->isIntegerTy(32)&&!L->getType()->isIntegerTy(64)))
        refuse("unsupported generic shared-pointer load width");
      global(M,"cpu_shared_memory",PointerType::getUnqual(M.getContext()));
      IRBuilder<> B(L);Value* V=B.CreatePtrToInt(get(B,M,"cpu_shared_memory"),B.getInt64Ty());
      if(offset==28)V=B.CreateLShr(V,32);
      if(L->getType()->isIntegerTy(32))V=B.CreateTrunc(V,B.getInt32Ty());
      L->replaceAllUsesWith(V);L->eraseFromParent();
      F.addFnAttr("cpu.generic.shared.pointer");continue;
    }
    if(!L->getType()->isIntegerTy(32))refuse("special register has non-i32 type");
    IRBuilder<> B(L);Value* V=nullptr;
    if(uniform.count(offset))V=get(B,M,uniform[offset]);
    else if(offset==44)V=B.CreateURem(tid(B,M),get(B,M,"block_size_x"));
    else if(offset==48)V=B.CreateURem(B.CreateUDiv(tid(B,M),get(B,M,"block_size_x")),get(B,M,"block_size_y"));
    else if(offset==52)V=B.CreateUDiv(tid(B,M),B.CreateMul(get(B,M,"block_size_x"),get(B,M,"block_size_y")));
    else if(offset==56)V=B.CreateAnd(tid(B,M),B.getInt32(31));
    else refuse(Twine("unsupported CuLifter special-register offset ")+Twine(offset));
    L->replaceAllUsesWith(V);L->eraseFromParent();
  }
  std::vector<CallInst*> calls;for(auto& I:instructions(F))if(auto* C=dyn_cast<CallInst>(&I))calls.push_back(C);
  for(auto* C:calls){auto* Callee=C->getCalledFunction();if(!Callee)continue;auto N=Callee->getName();
    if(!N.starts_with("llvm.nvvm.read.ptx.sreg."))continue;
    IRBuilder<> B(C);Value* V=nullptr;
    for(auto pair:std::map<std::string,std::string>{{"ntid.x","block_size_x"},{"ntid.y","block_size_y"},{"ntid.z","block_size_z"},
       {"nctaid.x","grid_size_x"},{"nctaid.y","grid_size_y"},{"nctaid.z","grid_size_z"},
       {"ctaid.x","block_index_x"},{"ctaid.y","block_index_y"},{"ctaid.z","block_index_z"}})
      if(N=="llvm.nvvm.read.ptx.sreg."+pair.first)V=get(B,M,pair.second);
    if(N.ends_with(".tid.x"))V=B.CreateURem(tid(B,M),get(B,M,"block_size_x"));
    if(N.ends_with(".tid.y"))V=B.CreateURem(B.CreateUDiv(tid(B,M),get(B,M,"block_size_x")),get(B,M,"block_size_y"));
    if(N.ends_with(".tid.z"))V=B.CreateUDiv(tid(B,M),B.CreateMul(get(B,M,"block_size_x"),get(B,M,"block_size_y")));
    if(N.ends_with(".laneid"))V=B.CreateAnd(tid(B,M),B.getInt32(31));
    if(!V)refuse(Twine("unsupported NVVM special register ")+N);
    C->replaceAllUsesWith(V);C->eraseFromParent();
  }
}
// Adaptation of CuPBoP warp_func.cpp store/barrier/read lowering. Each site has
// block-local storage; full-warp butterfly semantics are explicitly guarded.
static void lowerCollectives(Module& M,Function& F){
  std::vector<CallInst*> calls;for(auto& I:instructions(F))if(auto* C=dyn_cast<CallInst>(&I))calls.push_back(C);
  AllocaInst* partnerScratch=nullptr;
  for(auto* C:calls){auto* CF=C->getCalledFunction();if(!CF)refuse("indirect call or inline assembly");auto N=CF->getName();
    if(cpu_fp16::lower(M,F,C,BlockSize))continue;
    if(tensorName(N)||N=="barrier0_and"||N=="barrier0_or"||N=="barrier0_popc"){
      bool mma=N=="hmma1688_tf32_warp",ldsm=N=="ldsm_m88_warp",warp=mma||ldsm;
      if(warp&&BlockSize%32)refuse("matrix collective requires full warps");
      unsigned fields=mma?10:ldsm?4:1;
      auto* T=mma?Type::getFloatTy(M.getContext()):Type::getInt32Ty(M.getContext());
      IRBuilder<> E(&*F.getEntryBlock().getFirstInsertionPt());
      auto* A=E.CreateAlloca(T,E.getInt32(BlockSize*fields),"cpu.collective.fragments");
      A->setMetadata("cpu.block.shared",MDNode::get(M.getContext(),{}));
      IRBuilder<> B(C);
      auto sync=[&]{if(warp)marker(B,M);else B.CreateCall(M.getOrInsertFunction("llvm.nvvm.barrier0",B.getVoidTy()));};
      sync();
      auto* index=tid(B,M);auto* local=lane(B,M);auto* base=warpBase(B,M);
      for(unsigned i=0;i<fields;++i)B.CreateStore(C->getArgOperand(i+(warp?1:0)),B.CreateGEP(T,A,B.CreateAdd(B.CreateMul(index,B.getInt32(fields)),B.getInt32(i))));
      sync();
      auto read=[&](Value* peer,Value* field)->Value*{
        return B.CreateLoad(T,B.CreateGEP(T,A,B.CreateAdd(B.CreateMul(peer,B.getInt32(fields)),field)));
      };
      if(mma){
        auto* group=B.CreateLShr(local,2);auto* thread=B.CreateAnd(local,B.getInt32(3));
        auto round=[&](Value* value)->Value*{return B.CreateBitCast(B.CreateAnd(B.CreateBitCast(value,B.getInt32Ty()),B.getInt32(0xffffe000u)),B.getFloatTy());};
        for(unsigned out=0;out<4;++out){
          auto* column=B.CreateAdd(B.CreateMul(thread,B.getInt32(2)),B.getInt32(out%2));
          Value* result=C->getArgOperand(7+out);
          for(unsigned k=0;k<8;++k){
            auto* alane=B.CreateAdd(base,B.CreateAdd(B.CreateMul(group,B.getInt32(4)),B.getInt32(k%4)));
            auto* blane=B.CreateAdd(base,B.CreateAdd(B.CreateMul(column,B.getInt32(4)),B.getInt32(k%4)));
            auto* av=round(read(alane,B.getInt32((out/2)+(k/4)*2)));
            auto* bv=round(read(blane,B.getInt32(4+k/4)));
            result=B.CreateCall(Intrinsic::getDeclaration(&M,Intrinsic::fma,{B.getFloatTy()}),{av,bv,result});
          }
          B.CreateStore(result,B.CreateGEP(B.getFloatTy(),C->getArgOperand(0),B.getInt32(out)));
        }
      }else if(ldsm){
        auto count=constant(C->getArgOperand(5),"matrix count"),transpose=constant(C->getArgOperand(6),"matrix transpose");
        if((count!=1&&count!=2&&count!=4)||transpose>1)refuse("invalid matrix-load shape");
        auto* group=B.CreateLShr(local,2);auto* thread=B.CreateAnd(local,B.getInt32(3));
        for(unsigned matrix=0;matrix<count;++matrix){Value* value;
          if(!transpose)value=read(B.CreateAdd(base,B.CreateAdd(B.getInt32(8*matrix),group)),thread);
          else{
            auto* row=B.CreateAdd(base,B.CreateAdd(B.getInt32(8*matrix),B.CreateMul(thread,B.getInt32(2))));
            auto* low=read(row,B.CreateLShr(group,1));auto* high=read(B.CreateAdd(row,B.getInt32(1)),B.CreateLShr(group,1));
            auto* shift=B.CreateMul(B.CreateAnd(group,B.getInt32(1)),B.getInt32(16));
            value=B.CreateOr(B.CreateAnd(B.CreateLShr(low,shift),B.getInt32(65535)),B.CreateShl(B.CreateAnd(B.CreateLShr(high,shift),B.getInt32(65535)),16));
          }
          B.CreateStore(value,B.CreateGEP(B.getInt32Ty(),C->getArgOperand(0),B.getInt32(matrix)));
        }
      }else{
        Value* value=B.getInt32(N=="barrier0_and"?1:0);
        for(unsigned i=0;i<BlockSize;++i){auto* bit=B.CreateZExt(B.CreateICmpNE(read(B.getInt32(i),B.getInt32(0)),B.getInt32(0)),B.getInt32Ty());
          value=N=="barrier0_and"?B.CreateAnd(value,bit):N=="barrier0_or"?B.CreateOr(value,bit):B.CreateAdd(value,bit);}
        C->replaceAllUsesWith(value);
      }
      sync();C->eraseFromParent();continue;
    }
    bool shfl=shuffleName(N);
    bool sum=N=="region_sum";
    if(!shfl&&!sum)continue;
    if(BlockSize%32)refuse("full-warp collective requires block size divisible by 32");
    if(shfl&&constant(C->getArgOperand(0),"shuffle mask")!=0xffffffffu)refuse("only full-mask shuffle supported");
    uint32_t clamp=shfl?constant(C->getArgOperand(3),"shuffle clamp"):31;
    if(shfl&&!N.contains("idx")&&clamp!=31)refuse("segmented non-indexed shuffle is unsupported");
    // IDX consumes the low five bits, including for a register source. Unlike
    // UP/DOWN/BFLY it does not require a statically bounded distance.
    if(shfl&&!N.contains("idx")&&constant(C->getArgOperand(2),"shuffle distance")>31)refuse("shuffle distance outside warp");
    IRBuilder<> E(&*F.getEntryBlock().getFirstInsertionPt());
    Type* T=C->getType();auto* A=E.CreateAlloca(T,E.getInt32(BlockSize),"cpu.collective.values");
    A->setMetadata("cpu.block.shared",MDNode::get(M.getContext(),{}));
    IRBuilder<> B(C);marker(B,M);
    auto* V=C->getArgOperand(sum?0:1);
    B.CreateStore(V,B.CreateGEP(T,A,tid(B,M)));
    marker(B,M);
    Value* R=nullptr;
    if(shfl){Value* peer=nullptr;auto* index=lane(B,M);auto* distance=C->getArgOperand(2);
      if(N.contains("bfly"))peer=B.CreateXor(index,distance);
      else if(N.contains("down")){auto* next=B.CreateAdd(index,distance);peer=B.CreateSelect(B.CreateICmpULT(next,B.getInt32(32)),next,index);}
      else if(N.contains("up")){auto* next=B.CreateSub(index,distance);peer=B.CreateSelect(B.CreateICmpUGE(index,distance),next,index);}
      else if(N.contains("idx")){
        // PTX shfl.idx: bval=b[4:0], min=lane&c[12:8],
        // max=min|(c[4:0]&~c[12:8]), j=min|(bval&~c[12:8]).
        // An out-of-range source returns this lane's original value.
        auto segment=(clamp>>8)&31u;
        auto* first=B.CreateAnd(index,B.getInt32(segment));
        auto* last=B.CreateOr(first,B.getInt32((clamp&31u)&~segment));
        auto* source=B.CreateOr(first,B.CreateAnd(distance,B.getInt32(31u&~segment)));
        peer=B.CreateSelect(B.CreateICmpULE(source,last),source,index);
      }
      else refuse("unrecognized shuffle mode");
      auto* src=B.CreateAdd(warpBase(B,M),peer);
      R=B.CreateLoad(T,B.CreateGEP(T,A,src),"cpu.shuffle");
    }else{
      auto* Results=E.CreateAlloca(T,E.getInt32((BlockSize+31)/32),"cpu.collective.results");
      Results->setMetadata("cpu.block.shared",MDNode::get(M.getContext(),{}));
      // One owner computes the reduction; other lanes read it only in the next phase.
      auto* Then=SplitBlockAndInsertIfThen(B.CreateICmpEQ(lane(B,M),B.getInt32(0)),C,false);
      IRBuilder<> O(Then);auto* Begin=O.CreateGEP(T,A,warpBase(O,M));
      auto Reduce=M.getOrInsertFunction("cpu_region_sasum",T,O.getInt32Ty(),O.getPtrTy());
      auto* S=O.CreateCall(Reduce,{O.getInt32(32),Begin});
      O.CreateStore(S,O.CreateGEP(T,Results,get(O,M,"inter_warp_index")));
      B.SetInsertPoint(C);marker(B,M);
      R=B.CreateLoad(T,B.CreateGEP(T,Results,get(B,M,"inter_warp_index")),"cpu.sum");
      if(constant(C->getArgOperand(1),"reduction site")==0x70000000u)partnerScratch=A;
    }
    // A region boundary prevents future loop iterations from overwriting inputs
    // before every lane has consumed them.
    marker(B,M);C->replaceAllUsesWith(R);C->eraseFromParent();
  }
  std::vector<CallInst*> partners;
  for(auto& I:instructions(F))if(auto* C=dyn_cast<CallInst>(&I))if(C->getCalledFunction()&&C->getCalledFunction()->getName()=="region_sum_partner")partners.push_back(C);
  for(auto* C:partners){if(!partnerScratch)refuse("partner collective without first reduction");IRBuilder<> B(C);
    auto Callee=M.getOrInsertFunction("cpu_region_partner",B.getFloatTy(),B.getPtrTy(),B.getInt32Ty());
    auto* V=B.CreateCall(Callee,{B.CreateGEP(B.getFloatTy(),partnerScratch,warpBase(B,M)),lane(B,M)});
    C->replaceAllUsesWith(V);C->eraseFromParent();
  }
}
static void auditInput(Module& M,Function& F){
  if(!F.arg_empty())refuse("entry arguments require an ABI adapter; CuLifter entries use const_mem");
  for(auto& G:M.globals()){
    if(G.getAddressSpace()!=0)refuse("nonzero global address space requires memory-hierarchy lowering");
    if(G.isThreadLocal()&&G.getName()!="local_mem"&&G.getName()!="const_mem")refuse(Twine("unregistered lane-private global: ")+G.getName());
  }
  for(auto& I:instructions(F)){
    if(isa<InvokeInst>(I)||isa<CallBrInst>(I)||isa<IndirectBrInst>(I))refuse("unsupported nonlocal control flow");
    if(auto* C=dyn_cast<CallInst>(&I)){
      auto* CF=C->getCalledFunction();if(!CF)refuse("indirect call or inline assembly");auto N=CF->getName();
      if(CF->getIntrinsicID()!=Intrinsic::not_intrinsic&&!CF->isTargetIntrinsic())continue;
      if(N=="llvm.nvvm.bar.warp.sync"&&((BlockSize%32&&!C->getMetadata("cpu.warp.memory.order"))||constant(C->getArgOperand(0),"warp-sync mask")!=0xffffffffu))refuse("only full-warp synchronization is supported");
      if(N.starts_with("llvm.nvvm.read.ptx.sreg.")||N=="llvm.nvvm.barrier0"||N=="llvm.nvvm.bar.warp.sync"||N=="llvm.nvvm.shfl.sync.bfly.i32"||N=="llvm.nvvm.shfl.sync.bfly.f32")continue;
      if(shuffleName(N)||tensorName(N)||N=="barrier0_and"||N=="barrier0_or"||N=="barrier0_popc"||N=="region_sum"||N=="region_sum_partner"||N=="region_affine"||N=="rcp_f32"||N=="rsqrt_f32"||N=="fma_rm_f32"||N=="fma_rp_f32"||N=="fma_rz_f32")continue;
      refuse(Twine("unregistered call: ")+N);
    }
  }
}
static bool hasOperand(Value* V,GlobalVariable* G){
  if(V==G)return true;
  if(auto* E=dyn_cast<ConstantExpr>(V))for(auto& O:E->operands())if(hasOperand(O,G))return true;
  return false;
}
static Value* privateOperand(Value* V,Instruction* Before,GlobalVariable* G,Module& M){
  if(V==G){IRBuilder<> B(Before);auto* Base=get(B,M,"cpu_local_memory");
    return B.CreateGEP(B.getInt8Ty(),Base,B.CreateMul(B.CreateZExt(tid(B,M),B.getInt64Ty()),B.getInt64(32768)),"cpu.private.base");}
  if(auto* E=dyn_cast<ConstantExpr>(V))if(hasOperand(E,G)){
    auto* I=E->getAsInstruction();I->insertBefore(Before);
    for(unsigned k=0;k<I->getNumOperands();k++)I->setOperand(k,privateOperand(I->getOperand(k),I,G,M));
    return I;
  }
  return V;
}
static bool normalizePrivateMemory(Module& M,Function& F){
  auto* G=M.getGlobalVariable("local_mem");if(!G||G->use_empty())return false;
  global(M,"cpu_local_memory",PointerType::getUnqual(M.getContext()));
  std::vector<Instruction*> original;for(auto& I:instructions(F))original.push_back(&I);
  bool used=false;
  for(auto* I:original)for(unsigned k=0;k<I->getNumOperands();k++){
    auto* V=I->getOperand(k);if(hasOperand(V,G)){used=true;I->setOperand(k,privateOperand(V,I,G,M));}
  }
  return used;
}
static bool normalizeSharedMemory(Module& M,Function& F){
  auto* G=M.getGlobalVariable("shared_mem");if(!G||G->use_empty())return false;
  global(M,"cpu_shared_memory",PointerType::getUnqual(M.getContext()));
  std::function<Value*(Value*,Instruction*)> replace=[&](Value* V,Instruction* Before)->Value*{
    if(V==G){IRBuilder<> B(Before);return get(B,M,"cpu_shared_memory");}
    if(auto* E=dyn_cast<ConstantExpr>(V))if(hasOperand(E,G)){auto* I=E->getAsInstruction();I->insertBefore(Before);
      for(unsigned k=0;k<I->getNumOperands();++k)I->setOperand(k,replace(I->getOperand(k),I));return I;}
    return V;
  };
  std::vector<Instruction*> original;for(auto& I:instructions(F))original.push_back(&I);
  for(auto* I:original)for(unsigned k=0;k<I->getNumOperands();++k)I->setOperand(k,replace(I->getOperand(k),I));
  return true;
}
static void postAudit(Module& M,Function& F){
  for(auto& I:instructions(F))if(auto* C=dyn_cast<CallInst>(&I))if(auto* CF=C->getCalledFunction()){
    auto N=CF->getName();if(N.starts_with("llvm.nvvm.")||N=="shfl_bfly_f32"||N=="region_sum"||N=="region_sum_partner"||N.starts_with("vx_"))refuse(Twine("unlowered collective or target helper: ")+N);
  }
  if(verifyModule(M,&errs()))refuse("LLVM verifier rejected transformed module");
}
static void rematerializeLogicalIndices(Module& M,Function& F){
  // Lane/warp identifiers are implicit execution coordinates, not captured
  // program state. Read them at each use so no source SSA load crosses a
  // collective boundary. Saving loop-control loads during the outer collapse
  // would instead snapshot the serializer's own changing induction variable.
  std::vector<LoadInst*> loads;
  for(auto& I:instructions(F))if(auto* L=dyn_cast<LoadInst>(&I))
    if(L->getPointerOperand()==M.getNamedGlobal("intra_warp_index") ||
       L->getPointerOperand()==M.getNamedGlobal("inter_warp_index"))loads.push_back(L);
  for(auto* L:loads){
    while(!L->use_empty()){
      auto& U=*L->use_begin();auto* User=cast<Instruction>(U.getUser());
      if(isa<PHINode>(User))refuse("logical index rematerialization requires demoted PHIs");
      auto* Copy=cast<LoadInst>(L->clone());Copy->insertBefore(User);U.set(Copy);
    }
    L->eraseFromParent();
  }
}
void preserve_cpu_barrier_free_branches(Function& F){
  // Synthetic region separators must not pull a lane-varying predicate out
  // of a barrier-free diamond. Keep that diamond inside the lane loop.
  PostDominatorTree PDT(F);SmallPtrSet<CallInst*,32> erase;
  for(auto& Head:F){
    auto* Br=dyn_cast<BranchInst>(Head.getTerminator());
    if(!Br||!Br->isConditional())continue;
    auto* Node=PDT.getNode(&Head);if(!Node||!Node->getIDom())continue;
    auto* Merge=Node->getIDom()->getBlock();if(!Merge)continue;
    SmallPtrSet<BasicBlock*,32> seen;SmallVector<BasicBlock*,32> pending{&Head};
    SmallVector<CallInst*,16> candidates;bool real=false;
    while(!pending.empty()){
      auto* B=pending.pop_back_val();if(B==Merge||!seen.insert(B).second)continue;
      for(auto& I:*B)if(auto* C=dyn_cast<CallInst>(&I))if(auto* CF=C->getCalledFunction())
        if(CF->getName().starts_with("llvm.nvvm.bar")){
          // A real collective before the branch is outside the diamond.
          // Counting it would retain synthetic cuts on otherwise ordinary
          // predicated stores, producing overlapping lane-loop regions.
          if(!C->getMetadata("cpu.synthetic.boundary")){if(B!=&Head)real=true;}
          else if(B!=&F.getEntryBlock()&&!isa<ReturnInst>(B->getTerminator()))candidates.push_back(C);
        }
      for(auto* S:successors(B)){
        // This is a loop, not an acyclic diamond; keep its phase boundaries.
        if(S==&Head)real=true;
        pending.push_back(S);
      }
    }
    if(!real)for(auto* C:candidates)erase.insert(C);
  }
  for(auto* C:erase)C->eraseFromParent();
}
static bool boundary(BasicBlock* B){
  for(auto& I:*B)if(auto* C=dyn_cast<CallInst>(&I))if(auto* CF=C->getCalledFunction())
    if(CF->getName().starts_with("llvm.nvvm.bar"))return true;
  return false;
}
void normalize_cpu_barrier_forks(Function& F){
  // The backward region walk must not wrap the same predicate-computation
  // prefix once per outgoing phase. Isolate a fork when its arms encounter
  // different next boundaries. Barrier-free diamonds have identical frontiers
  // and remain ordinary lane control flow. Loop backedges are handled by the
  // loop-boundary pass, not as acyclic forks.
  DominatorTree DT(F);
  SmallVector<BasicBlock*,64> heads;
  for(auto& B:F)if(auto* Br=dyn_cast<BranchInst>(B.getTerminator()))
    if(Br->isConditional()&&!boundary(&B))heads.push_back(&B);
  for(auto* Head:heads){
    auto* Br=cast<BranchInst>(Head->getTerminator());
    std::set<BasicBlock*> frontiers[2];bool cyclic=false;
    for(unsigned arm=0;arm<2;++arm){
      SmallVector<BasicBlock*,32> pending{Br->getSuccessor(arm)};
      SmallPtrSet<BasicBlock*,32> seen;
      while(!pending.empty()){
        auto* B=pending.pop_back_val();
        if(B==Head||DT.dominates(B,Head)){cyclic=true;break;}
        if(!seen.insert(B).second)continue;
        if(boundary(B)){frontiers[arm].insert(B);continue;}
        for(auto* S:successors(B))pending.push_back(S);
      }
    }
    if(cyclic||frontiers[0].empty()||frontiers[1].empty()||frontiers[0]==frontiers[1])continue;
    bool cta=false;for(auto& group:frontiers)for(auto* B:group)cta|=has_block_barrier(B);
    auto* C=cta?CreateInterWarpBarrier(Br):CreateIntraWarpBarrier(Br);
    Head->splitBasicBlock(C,"cpu.fork.boundary");
  }
}
static bool normalizeBarrierJoinRound(Function& F){
  // CuPBoP's backward region walk requires one preceding boundary. At a
  // reconvergence, group incoming edges by that boundary so each arm's tail
  // has a distinct exit before the common continuation. A scalar store in
  // such a tail must not be silently left outside serialization.
  SmallVector<BasicBlock*,64> order;
  bool changed=false;
  for(auto* B:ReversePostOrderTraversal<Function*>(&F))order.push_back(B);
  for(auto* Join:order){
    if(pred_size(Join)<2||Join->getTerminator()->getMetadata("cpu.join.normalized"))continue;
    std::map<BasicBlock*,SmallVector<BasicBlock*,8>> groups;bool complete=true;
    for(auto* Pred:predecessors(Join)){
      SmallVector<BasicBlock*,16> pending{Pred};SmallPtrSet<BasicBlock*,32> seen,frontier;
      while(!pending.empty()){
        auto* B=pending.pop_back_val();if(B==Join||!seen.insert(B).second)continue;
        if(boundary(B)){frontier.insert(B);continue;}
        for(auto* P:predecessors(B))pending.push_back(P);
      }
      if(frontier.size()!=1){complete=false;break;}
      groups[*frontier.begin()].push_back(Pred);
    }
    if(!complete||groups.size()<2)continue;
    changed=true;
    // Use source order rather than pointer order when constructing the CFG.
    SmallVector<BasicBlock*,8> frontiers;
    for(auto& B:F)if(groups.count(&B))frontiers.push_back(&B);
    bool cta=false;for(auto* B:frontiers)cta|=has_block_barrier(B);
    for(auto* B:frontiers){
      auto* Tail=SplitBlockPredecessors(Join,groups[B],".cpu.phase.tail",static_cast<DominatorTree*>(nullptr));
      auto* C=cta?CreateInterWarpBarrier(Tail->getTerminator()):CreateIntraWarpBarrier(Tail->getTerminator());
      Tail->splitBasicBlock(C,"cpu.join.boundary");
    }
    if(!boundary(Join)){
      auto* First=&*Join->getFirstInsertionPt();
      auto* C=cta?CreateInterWarpBarrier(First):CreateIntraWarpBarrier(First);
      auto* Sync=Join->splitBasicBlock(C,"cpu.join.boundary");
      if(!First->isTerminator())Sync->splitBasicBlock(First,"cpu.phase.continue");
    }
    Join->getTerminator()->setMetadata("cpu.join.normalized",MDNode::get(F.getContext(),{}));
  }
  return changed;
}
void normalize_cpu_barrier_joins(Function& F){
  // Dead lifted branches otherwise contribute an empty boundary frontier and
  // prevent normalization of a reachable join (including the kernel exit).
  removeUnreachableBlocks(F);
  // A join normalized for lanes may require another cut after warp markers
  // are removed. The visited mark belongs to this invocation, not the IR.
  for(auto& B:F)B.getTerminator()->setMetadata("cpu.join.normalized",nullptr);
  unsigned rounds=0;
  while(normalizeBarrierJoinRound(F))if(++rounds>64)refuse("barrier join normalization did not converge");
  for(auto& B:F)B.getTerminator()->setMetadata("cpu.join.normalized",nullptr);
}
int main(int argc,char** argv){cl::ParseCommandLineOptions(argc,argv);try{
  if(!BlockSize||BlockSize>1024)refuse("block-size must be 1..1024");
  if(!SharedBytes||SharedBytes>1048576)refuse("shared-memory-bytes must be 1..1048576");
  LLVMContext C;SMDiagnostic Err;auto M=parseIRFile(Input,Err,C);if(!M){Err.print(argv[0],errs());return 2;}
  auto* F=M->getFunction(Kernel);if(!F||F->isDeclaration())refuse("selected kernel definition missing");
  if(F->hasFnAttribute("cpu.coarsened"))refuse("kernel is already coarsened");
  if(verifyModule(*M,&errs()))refuse("input LLVM verification failed");
  normalizeHelpers(*M,*F);auditInput(*M,*F);
  // This artifact contains one selected entry point. Other input kernels remain
  // declarations, never untransformed executable code hidden in the output.
  for(auto& Other:*M)if(&Other!=F&&!Other.isDeclaration())Other.deleteBody();
  M->setTargetTriple(Target);M->setDataLayout("e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128");
  if(StringRef(Target).starts_with("aarch64"))M->setDataLayout("e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128");
  F->addFnAttr("cpu.coarsen.kernel");F->addFnAttr("cpu.block_size",std::to_string(BlockSize));
  // One exit boundary lets lane-local early exits converge before advancing
  // the lane loop instead of escaping the collapsed block function.
  std::vector<ReturnInst*> returns;for(auto& BB:*F)if(auto* R=dyn_cast<ReturnInst>(BB.getTerminator()))returns.push_back(R);
  if(!F->getReturnType()->isVoidTy())refuse("kernel must return void");
  if(returns.size()>1){auto* Exit=BasicBlock::Create(C,"cpu.kernel.exit",F);ReturnInst::Create(C,Exit);
    for(auto* R:returns){BranchInst::Create(Exit,R);R->eraseFromParent();}}
  prepareGlobals(*M);normalizeRegisters(*M,*F);simplifyScalarIR(*F);
  // CuPBoP's init_block also demotes PHIs before region discovery. Keep that
  // prerequisite while using LLVM's edge-correct utility.
  std::vector<PHINode*> phis;for(auto& I:instructions(F))if(auto* P=dyn_cast<PHINode>(&I))phis.push_back(P);
  for(auto* P:phis)DemotePHIToStack(P);
  bool privateMemory=normalizePrivateMemory(*M,*F);
  bool sharedMemory=normalizeSharedMemory(*M,*F)||F->hasFnAttribute("cpu.generic.shared.pointer");
  lowerCollectives(*M,*F);rematerializeLogicalIndices(*M,*F);
  // Isolate synchronization sites before static predication so every value
  // crossing a masked phase is preserved as lane-private state.
  split_block_by_sync(F);
  if(cpu_masks::run(*M,*F)) {
    lowerCollectives(*M,*F);
    // Collective scratch has semantic ownership metadata. InstCombine may
    // replace an alloca with an array alloca without copying that metadata,
    // making shared scratch look lane-private. After ownership assignment,
    // only canonicalize loop structure until the storage has been lowered.
    simplifyScalarIR(*F,false);
    std::vector<PHINode*> maskPhis;for(auto& I:instructions(*F))if(auto* P=dyn_cast<PHINode>(&I))maskPhis.push_back(P);
    for(auto* P:maskPhis)DemotePHIToStack(P);
    rematerializeLogicalIndices(*M,*F);
  }
  dump(*M,".normalized");
  setenv("VORTEX_SCHEDULE_FLAG","0",1);
  insert_sync(M.get());preserve_cpu_barrier_free_branches(*F);split_block_by_sync(M.get());normalize_cpu_barrier_forks(*F);normalize_cpu_barrier_joins(*F);dump(*M,".regions");
  for(auto& I:instructions(*F))I.setMetadata("cpu.lane.value",MDNode::get(C,{}));
  insert_warp_loop(M.get());dump(*M,".collapsed");
  if(privateMemory){IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());
    B.CreateCall(M->getOrInsertFunction("cpu_prepare_local_memory",B.getVoidTy(),B.getInt64Ty()),{B.getInt64(uint64_t(BlockSize)*32768)});}
  if(sharedMemory){IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());
    B.CreateCall(M->getOrInsertFunction("cpu_prepare_shared_memory",B.getVoidTy(),B.getInt64Ty()),{B.getInt64(SharedBytes)});}
  postAudit(*M,*F);
  if(!Rename.empty())F->setName(Rename);
  F->removeFnAttr("cpu.coarsen.kernel");F->addFnAttr("cpu.coarsened","cupbop-vortex");F->addFnAttr("cpu.block_size",std::to_string(BlockSize));
  std::error_code E;raw_fd_ostream O(Output,E);if(E)refuse(E.message());M->print(O,nullptr);
  outs()<<"{\"status\":\"ADMITS\",\"kernel\":\""<<F->getName()<<"\",\"block_size\":"<<BlockSize<<",\"mapping\":\"block-to-worker\"}\n";return 0;
 }catch(const std::exception& E){errs()<<"RECOGNISES-DOES-NOT-ADMIT: "<<E.what()<<"\n";return 2;}}
