// Derived from CuPBoP_Vortex ee05a48d79cbae11351e9c8eb711f286c1206402; see ../LICENSE and provenance.json.
#ifndef __NVVM2x86_INSERT_WARP_LOOP__
#define __NVVM2x86_INSERT_WARP_LOOP__

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

void insert_warp_loop(llvm::Module *M);

#endif