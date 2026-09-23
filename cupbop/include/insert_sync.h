// Derived from CuPBoP_Vortex ee05a48d79cbae11351e9c8eb711f286c1206402; see ../LICENSE and provenance.json.
#ifndef __NVVM2x86_INSERT_SYNC__
#define __NVVM2x86_INSERT_SYNC__

#include "llvm/IR/Function.h"

// insert extra barrier
void insert_sync(llvm::Module *M);

#endif
