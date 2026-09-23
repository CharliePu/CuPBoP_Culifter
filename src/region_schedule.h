#pragma once
#include "llvm/IR/Function.h"
#include <string>

namespace cpu_schedule {
// Analysis precedes storage lowering. A phase-free region keeps its original
// SSA graph; only the phased plan requires CuPBoP's cross-phase lane storage.
enum class Kind { WholeLane, Phased };
struct Plan {
  Kind kind;
  std::string reason;
};
Plan analyze(const llvm::Function &function);
void emitWholeLane(llvm::Function &function, unsigned blockSize);
unsigned lowerScalarMath(llvm::Function &function);
}
