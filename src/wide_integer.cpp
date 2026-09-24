#include "wide_integer.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/KnownBits.h"
#include <vector>

using namespace llvm;
using namespace llvm::PatternMatch;
namespace cpu_schedule {
namespace {
bool shiftIs(Value *V, unsigned Op, Value *&X, unsigned Amount) {
  auto *B = dyn_cast<BinaryOperator>(V);
  auto *K = B ? dyn_cast<ConstantInt>(B->getOperand(1)) : nullptr;
  if (!B || B->getOpcode() != Op || !K || K->getValue() != Amount) return false;
  X = B->getOperand(0);
  return true;
}

bool unpack(Value *V, Value *&Lo, Value *&Hi, unsigned &Width) {
  auto *B = dyn_cast<BinaryOperator>(V);
  if (!B || B->getOpcode() != Instruction::Or || !V->getType()->isIntegerTy()) return false;
  auto Bits = V->getType()->getIntegerBitWidth();
  if (Bits < 4 || Bits > 64 || Bits % 2) return false;
  Width = Bits / 2;
  for (unsigned Swap = 0; Swap != 2; ++Swap) {
    Value *H = nullptr;
    auto *L = dyn_cast<ZExtInst>(B->getOperand(Swap));
    if (!L || !L->getSrcTy()->isIntegerTy(Width) ||
        !shiftIs(B->getOperand(1-Swap), Instruction::Shl, H, Width)) continue;
    auto *Z = dyn_cast<ZExtInst>(H);
    Lo = L->getOperand(0);
    Hi = Z && Z->getSrcTy()->isIntegerTy(Width) ? Z->getOperand(0) : H;
    return true;
  }
  return false;
}

Value *join(IRBuilder<> &B, Value *Lo, Value *Hi) {
  auto W = Lo->getType()->getIntegerBitWidth();
  auto *T = IntegerType::get(B.getContext(), 2*W);
  auto *HighWord=B.CreateZExtOrTrunc(Hi,Lo->getType());
  return B.CreateOr(B.CreateZExt(Lo,T), B.CreateShl(B.CreateZExt(HighWord,T), W), "cpu.wide.join");
}

void addTerms(Value *V, SmallVectorImpl<Value *> &Terms) {
  if (Terms.size() > 8) return;
  auto *B = dyn_cast<BinaryOperator>(V);
  if (B && B->getOpcode() == Instruction::Add) {
    addTerms(B->getOperand(0), Terms); addTerms(B->getOperand(1), Terms);
  } else Terms.push_back(V);
}

// Prove equality modulo 2^W without materializing speculative computations.
// The depth cap only limits matching, never changes the source on uncertainty.
bool sameLow(Value *N, Value *Wide, unsigned W, const DataLayout &DL, unsigned Depth=0) {
  if (Depth > 16) return false;
  if (N == Wide) return true;
  if (auto *T = dyn_cast<TruncInst>(N))
    if (T->getOperand(0) == Wide) return true;
  if (auto *K = dyn_cast<ConstantInt>(N)) {
    auto Bits = computeKnownBits(Wide, DL);
    auto Mask = APInt::getLowBitsSet(Bits.getBitWidth(), W);
    if (((Bits.Zero | Bits.One) & Mask) == Mask)
      return Bits.One.trunc(W) == K->getValue();
  }
  if (auto *C = dyn_cast<CastInst>(Wide))
    if ((C->getOpcode()==Instruction::ZExt || C->getOpcode()==Instruction::SExt ||
         C->getOpcode()==Instruction::Trunc) && C->getSrcTy()->isIntegerTy() &&
        C->getSrcTy()->getIntegerBitWidth() >= W)
      return sameLow(N, C->getOperand(0), W, DL, Depth+1);
  auto *B = dyn_cast<BinaryOperator>(Wide);
  if (!B) return false;
  if (B->getOpcode()==Instruction::Or || B->getOpcode()==Instruction::Add) {
    for (unsigned I=0; I!=2; ++I)
      if (computeKnownBits(B->getOperand(I),DL).countMinTrailingZeros() >= W)
        if (sameLow(N,B->getOperand(1-I),W,DL,Depth+1)) return true;
  }
  auto *A = dyn_cast<BinaryOperator>(N);
  if (!A || A->getOpcode()!=B->getOpcode()) return false;
  auto Op = A->getOpcode();
  if (Op!=Instruction::Add && Op!=Instruction::Mul && Op!=Instruction::Or &&
      Op!=Instruction::And && Op!=Instruction::Xor && Op!=Instruction::Shl) return false;
  if (Op==Instruction::Shl) {
    auto *AK=dyn_cast<ConstantInt>(A->getOperand(1));
    auto *BK=dyn_cast<ConstantInt>(B->getOperand(1));
    return AK && BK && AK->getZExtValue()==BK->getZExtValue() && AK->getZExtValue()<W &&
           sameLow(A->getOperand(0),B->getOperand(0),W,DL,Depth+1);
  }
  for (unsigned Swap=0; Swap!=2; ++Swap)
    if (sameLow(A->getOperand(0),B->getOperand(Swap),W,DL,Depth+1) &&
        sameLow(A->getOperand(1),B->getOperand(1-Swap),W,DL,Depth+1)) return true;
  return false;
}

Value *recoverJoin(Instruction &I, Value *Lo, Value *Hi, unsigned W) {
  IRBuilder<> B(&I);
  const auto &DL = I.getModule()->getDataLayout();
  // join(low(X), high(X)) == X, even when low(X) was independently simplified.
  {
    auto *T=dyn_cast<TruncInst>(Hi);
    Value *X=nullptr;
    if (shiftIs(T?T->getOperand(0):Hi,Instruction::LShr,X,W) &&
        X->getType()==I.getType() && sameLow(Lo,X,W,DL)) return X;
  }
  // join(x, x >>signed (W-1)) == sext(x).
  Value *X=nullptr;
  if (shiftIs(Hi,Instruction::AShr,X,W-1) && X==Lo)
    return B.CreateSExt(Lo,I.getType(),"cpu.wide.signed");
  auto *LowAdd=dyn_cast<BinaryOperator>(Lo);
  if (!LowAdd || LowAdd->getOpcode()!=Instruction::Add) return nullptr;
  SmallVector<Value*,8> Terms; addTerms(Hi,Terms);
  if (Terms.size()<2 || Terms.size()>3) return nullptr;
  for (unsigned K=0; K<Terms.size(); ++K) {
    auto *Z=dyn_cast<ZExtInst>(Terms[K]);
    auto *Cmp=Z ? dyn_cast<ICmpInst>(Z->getOperand(0)) : nullptr;
    if (!Cmp || Cmp->getPredicate()!=ICmpInst::ICMP_ULT || Cmp->getOperand(0)!=Lo) continue;
    auto *A=LowAdd->getOperand(0), *C=LowAdd->getOperand(1);
    if (Cmp->getOperand(1)!=A && Cmp->getOperand(1)!=C) continue;
    // The high-word add is modulo 2^W. Assigning its non-carry terms to
    // either low operand gives exactly the same full-width sum.
    SmallVector<Value*,2> Highs;
    for (unsigned J=0; J<Terms.size(); ++J) if (J!=K) Highs.push_back(Terms[J]);
    if (Highs.size()==1) Highs.push_back(ConstantInt::get(Lo->getType(),0));
    // Prefer a pairing that exposes existing high/low halves of one value.
    auto matchesHigh=[&](Value *H,Value *L) {
      auto *T=dyn_cast<TruncInst>(H);Value *V=nullptr;
      return shiftIs(T?T->getOperand(0):H,Instruction::LShr,V,W) && sameLow(L,V,W,DL);
    };
    if(matchesHigh(Highs[0],C) || matchesHigh(Highs[1],A)) std::swap(A,C);
    return B.CreateAdd(join(B,A,Highs[0]),join(B,C,Highs[1]),"cpu.wide.add");
  }
  return nullptr;
}
} // namespace

unsigned recoverWideIntegers(Function &F) {
  unsigned Count=0;
  // New joins expose inner carry chains. InstCombine between invocations
  // performs the ordinary cleanup; this pass supplies only modular identities.
  std::vector<Instruction*> Work;
  for (auto &I: instructions(F)) Work.push_back(&I);
  for (auto *I:Work) {
    // Factor a shared left shift so scaled word pairs become visible.
    // (a << (W+k)) | (b << k) == ((a << W) | b) << k, modulo the width.
    if(auto *O=dyn_cast<BinaryOperator>(I)) if(O->getOpcode()==Instruction::Or)
      for(unsigned Swap=0;Swap!=2;++Swap) {
        auto *A=dyn_cast<BinaryOperator>(O->getOperand(Swap));
        auto *B=dyn_cast<BinaryOperator>(O->getOperand(1-Swap));
        if(!A||!B||A->getOpcode()!=Instruction::Shl||B->getOpcode()!=Instruction::Shl||
           !I->getType()->isIntegerTy())continue;
        auto *AK=dyn_cast<ConstantInt>(A->getOperand(1));
        auto *BK=dyn_cast<ConstantInt>(B->getOperand(1));
        unsigned Bits=I->getType()->getIntegerBitWidth();
        if(!AK||!BK||BK->isZero()||AK->getValue().uge(Bits)||BK->getValue().uge(Bits)||
           AK->getZExtValue()!=BK->getZExtValue()+Bits/2||Bits%2)continue;
        IRBuilder<> Builder(I);
        auto *Inner=Builder.CreateOr(Builder.CreateShl(A->getOperand(0),Bits/2),B->getOperand(0));
        auto *V=Builder.CreateShl(Inner,BK->getZExtValue(),"cpu.wide.scaled");
        I->replaceAllUsesWith(V);I->eraseFromParent();++Count;I=nullptr;break;
      }
    if(!I)continue;
    Value *Lo=nullptr,*Hi=nullptr; unsigned W=0;
    if (!unpack(I,Lo,Hi,W)) continue;
    if (auto *V=recoverJoin(*I,Lo,Hi,W)) {
      I->replaceAllUsesWith(V); I->eraseFromParent(); ++Count;
    }
  }
  return Count;
}
}
