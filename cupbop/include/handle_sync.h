// Derived from CuPBoP_Vortex ee05a48d79cbae11351e9c8eb711f286c1206402; see ../LICENSE and provenance.json.
#ifndef __NVVM2x86_HANDLE_SYNC__
#define __NVVM2x86_HANDLE_SYNC__

#include "llvm/IR/Module.h"

using namespace llvm;

void split_block_by_sync(llvm::Module *M);
void split_block_by_sync(llvm::Function *F);

#endif
