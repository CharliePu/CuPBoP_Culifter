#pragma once
#include "llvm/IR/Function.h"

namespace cpu_schedule {
unsigned outlineReadOnlyLoops(llvm::Function &function);
}
