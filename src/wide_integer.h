#pragma once
#include "llvm/IR/Function.h"

namespace cpu_schedule {
// Recover modular wide arithmetic from low/high words and explicit carries.
// No pointer aliasing, signed-overflow, or floating-point assumptions are added.
unsigned recoverWideIntegers(llvm::Function &function);
}
