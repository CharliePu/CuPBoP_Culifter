// Derived from CuPBoP_Vortex ee05a48d79cbae11351e9c8eb711f286c1206402; see ../LICENSE and provenance.json.
#include "handle_sync.h"
#include "tool.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <vector>
#include <set>
#include <string>

#include "cg_sync.h"

using namespace llvm;


void split_block_by_sync(llvm::Function *F) {
  std::vector<llvm::Instruction *> sync_inst;
  llvm::SmallPtrSet<llvm::Instruction *,32> seen;
  auto addSync = [&](llvm::Instruction *instruction) {
    if (seen.insert(instruction).second) sync_inst.push_back(instruction);
  };
  // Jumping the first sync has been removed in LLVM 18 CPU CuPBoP
  // bool jump_first_sync = 1;
  for (Function::iterator b = F->begin(); b != F->end(); ++b) {
    BasicBlock *B = &(*b);
    for (BasicBlock::iterator i = B->begin(); i != B->end(); ++i) {
      Instruction *inst = &(*i);

      // first instruction is not always a sync, this
      // seems to be incorrect splitting?
      // if (jump_first_sync) {
      //   jump_first_sync = 0;
      //   Instruction *next_inst = &(*std::next(i));
      //   sync_inst.insert(next_inst);
      //   continue;
      // }
      llvm::CallInst *Call = llvm::dyn_cast<llvm::CallInst>(inst);
      if (Call) {
        if (Call->isInlineAsm() || !Call->getCalledFunction())
          continue;
        auto func_name = Call->getCalledFunction()->getName().str();
        if (func_name == "llvm.nvvm.barrier0" ||
            isWarpSync(func_name) ||
            func_name == "llvm.nvvm.barrier.sync" ||
            isCGSync(func_name) ||
            func_name == "cupbop.shfl.barrier") {
          //print whole block(b)
          
          if (cupbop_debug()) {
            printf("found barrier inst!\n");
            i->print(llvm::errs());
            b->print(errs());
          }
          addSync(Call);
          // we should also sync the next instruction
          // so that we can get a block with sync inst only.
          // Explicit predication guards must execute inside the phase loop.
          // Keep uniform collective-loop control with its marker; splitting
          // that scheduler branch into a lane phase can reset the loop index
          // on its backedge and prevent progress.
          Instruction *next_inst = &(*std::next(i));
          auto* NextBranch = dyn_cast<BranchInst>(next_inst);
          if (!next_inst->isTerminator() || (NextBranch && NextBranch->isConditional() && NextBranch->getMetadata("cpu.lane.guard")))
            addSync(next_inst);
        }
      }
    }
  }
  int _tmp = 0;
  for (auto inst : sync_inst) {
    if (cupbop_debug()) {
      printf("temp=%d\n", _tmp);
      printf("sync inst:");
      inst->print(errs());
      printf("block to be split:\n");
      inst->getParent()->print(errs());
    }
    inst->getParent()->splitBasicBlock(
        inst, "cpu.sync." + std::to_string(_tmp++));
  }
}

void split_block_by_sync(llvm::Module *M) {

  int schedule = 0;
  if (char *env = std::getenv("VORTEX_SCHEDULE_FLAG")) {
    schedule = std::stoi(std::string(env));
  }

  if (schedule == 0 || schedule == 1) {
    //printf("splitting block by sync starting\n");
    //printIR(M);
    for (Module::iterator i = M->begin(), e = M->end(); i != e; ++i) {
      Function *F = &(*i);
      if (isKernelFunction(M, F))
        split_block_by_sync(F);
    }
    //print the whole module 
    //printf("printing the whole module after splitting the block\n");
    //printIR(M);
  } else {
    if (cupbop_debug()) printf("no need to split block by sync\n");
  }
}
