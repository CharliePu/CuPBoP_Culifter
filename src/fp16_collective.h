// Deterministic lowering of CuLifter's packed m16n8k16 F16 collective.
// Include after the adapter's refuse/tid/lane/warpBase/marker helpers.
// Arithmetic and fragment mapping follow kernel_wrapper_75.h, not a vendor MMA.
#pragma once

namespace cpu_fp16 {
// Integer conversion reproduces half_bits_to_f32_, including subnormals and
// NaN payload bits. Do not let a target's half conversion choose FTZ behavior.
static llvm::Value* unpackInline(llvm::IRBuilder<>& B, llvm::Module& M,
                           llvm::Value* Word, unsigned High) {
  auto* H = B.CreateAnd(B.CreateLShr(Word, High ? 16 : 0), B.getInt32(65535));
  auto* Sign = B.CreateShl(B.CreateAnd(H, B.getInt32(32768)), 16);
  auto* Exp = B.CreateAnd(B.CreateLShr(H, 10), B.getInt32(31));
  auto* Mant = B.CreateAnd(H, B.getInt32(1023));
  auto* Leading = B.CreateCall(llvm::Intrinsic::getDeclaration(
      &M, llvm::Intrinsic::ctlz, {B.getInt32Ty()}), {Mant, B.getFalse()});
  // Mant == 0 gets shift 11 (valid), then the explicit signed-zero selection.
  auto* Shift = B.CreateSub(Leading, B.getInt32(21));
  auto* Sub = B.CreateOr(B.CreateShl(B.CreateSub(B.getInt32(113), Shift), 23),
      B.CreateShl(B.CreateAnd(B.CreateShl(Mant, Shift), B.getInt32(1023)), 13));
  auto* Normal = B.CreateOr(B.CreateShl(B.CreateAdd(Exp, B.getInt32(112)), 23),
                            B.CreateShl(Mant, 13));
  auto* Special = B.CreateOr(B.getInt32(0x7f800000), B.CreateShl(Mant, 13));
  auto* Bits = B.CreateSelect(B.CreateICmpEQ(Exp, B.getInt32(31)), Special, Normal);
  Bits = B.CreateSelect(B.CreateICmpEQ(Exp, B.getInt32(0)),
      B.CreateSelect(B.CreateICmpEQ(Mant, B.getInt32(0)), B.getInt32(0), Sub), Bits);
  return B.CreateBitCast(B.CreateOr(Sign, Bits), B.getFloatTy());
}

// Exact f32_to_half_bits_: nearest-even, gradual underflow, signed zero,
// overflow to infinity and canonical quiet NaN retaining its sign.
static llvm::Value* packInline(llvm::IRBuilder<>& B, llvm::Value* F) {
  auto* X = B.CreateBitCast(F, B.getInt32Ty());
  auto* Sign = B.CreateAnd(B.CreateLShr(X, 16), B.getInt32(0x8000));
  auto* Exp = B.CreateSub(B.CreateAnd(B.CreateLShr(X, 23), B.getInt32(255)), B.getInt32(112));
  auto* Mant = B.CreateAnd(X, B.getInt32(0x7fffff));
  auto round = [&](llvm::Value* H, llvm::Value* Rem, llvm::Value* Half) {
    auto* Up = B.CreateOr(B.CreateICmpUGT(Rem, Half),
        B.CreateAnd(B.CreateICmpEQ(Rem, Half), B.CreateICmpNE(B.CreateAnd(H, B.getInt32(1)), B.getInt32(0))));
    return B.CreateAdd(H, B.CreateZExt(Up, B.getInt32Ty()));
  };
  auto* Normal = B.CreateOr(Sign, B.CreateOr(B.CreateShl(Exp, 10), B.CreateLShr(Mant, 13)));
  Normal = round(Normal, B.CreateAnd(Mant, B.getInt32(8191)), B.getInt32(4096));
  // Clamp unused select arms so no shift creates LLVM poison.
  auto* SubExp = B.CreateSelect(B.CreateICmpSLT(Exp, B.getInt32(-10)), B.getInt32(-10), Exp);
  SubExp = B.CreateSelect(B.CreateICmpSGT(SubExp, B.getInt32(0)), B.getInt32(0), SubExp);
  auto* Shift = B.CreateSub(B.getInt32(14), SubExp);
  auto* Sig = B.CreateOr(Mant, B.getInt32(0x800000));
  auto* Sub = round(B.CreateLShr(Sig, Shift),
      B.CreateAnd(Sig, B.CreateSub(B.CreateShl(B.getInt32(1), Shift), B.getInt32(1))),
      B.CreateShl(B.getInt32(1), B.CreateSub(Shift, B.getInt32(1))));
  Sub = B.CreateSelect(B.CreateICmpSLT(Exp, B.getInt32(-10)), Sign, B.CreateOr(Sign, Sub));
  auto* IsNan = B.CreateAnd(B.CreateICmpEQ(Exp, B.getInt32(143)), B.CreateICmpNE(Mant, B.getInt32(0)));
  auto* Special = B.CreateOr(Sign, B.CreateOr(B.getInt32(0x7c00), B.CreateSelect(IsNan, B.getInt32(0x200), B.getInt32(0))));
  return B.CreateAnd(B.CreateSelect(B.CreateICmpSGE(Exp, B.getInt32(31)), Special,
      B.CreateSelect(B.CreateICmpSLE(Exp, B.getInt32(0)), Sub, Normal)), B.getInt32(65535));
}

// Keep exact scalar conversions as shared IR helpers. Expanding their dozens
// of integer operations for every matrix operand before context construction
// produced tens of megabytes of redundant lane state and compiler timeouts.
// These are deterministic arithmetic helpers, not a vendor-call replacement.
static llvm::Function* conversion(llvm::Module& M,bool ToFloat) {
  auto Name=ToFloat?"cpu.f16.unpack":"cpu.f16.pack";
  if(auto* F=M.getFunction(Name))return F;
  auto* I=llvm::Type::getInt32Ty(M.getContext());auto* T=llvm::Type::getFloatTy(M.getContext());
  auto* F=llvm::Function::Create(llvm::FunctionType::get(ToFloat?T:I,{ToFloat?I:T},false),llvm::GlobalValue::InternalLinkage,Name,M);
  F->addFnAttr(llvm::Attribute::NoInline);F->addFnAttr(llvm::Attribute::NoUnwind);
  F->addFnAttr(llvm::Attribute::WillReturn);F->addFnAttr(llvm::Attribute::NoSync);
  F->setDoesNotAccessMemory();
  auto* BB=llvm::BasicBlock::Create(M.getContext(),"entry",F);llvm::IRBuilder<> B(BB);
  B.CreateRet(ToFloat?unpackInline(B,M,F->getArg(0),0):packInline(B,F->getArg(0)));
  return F;
}
static llvm::Value* unpack(llvm::IRBuilder<>& B,llvm::Module& M,llvm::Value* Word,unsigned High) {
  if(High)Word=B.CreateLShr(Word,16);
  return B.CreateCall(conversion(M,true),{Word});
}
static llvm::Value* pack(llvm::IRBuilder<>& B,llvm::Value* Value) {
  return B.CreateCall(conversion(*B.GetInsertBlock()->getModule(),false),{Value});
}

template<class Body>
static void loop(llvm::IRBuilder<>& B,llvm::Function* F,unsigned Count,
                 llvm::StringRef Name,Body Emit) {
  auto& C=F->getContext();auto* Pre=B.GetInsertBlock();
  auto* Head=llvm::BasicBlock::Create(C,Name+".head",F);
  auto* Work=llvm::BasicBlock::Create(C,Name+".body",F);
  auto* Exit=llvm::BasicBlock::Create(C,Name+".exit",F);
  B.CreateBr(Head);B.SetInsertPoint(Head);
  auto* I=B.CreatePHI(B.getInt32Ty(),2,Name+".index");I->addIncoming(B.getInt32(0),Pre);
  B.CreateCondBr(B.CreateICmpULT(I,B.getInt32(Count)),Work,Exit);
  B.SetInsertPoint(Work);Emit(I);
  auto* Latch=B.GetInsertBlock();auto* Next=B.CreateAdd(I,B.getInt32(1));B.CreateBr(Head);I->addIncoming(Next,Latch);
  B.SetInsertPoint(Exit);
}

// One deterministic CPU function implements one full-warp tensor instruction.
// Decode each shared fragment once, retain ascending-K FP32 mul/add, then pack
// each output at the original FP16 boundary. No provider or workload code is
// involved. The explicit matrix loop also preserves a useful post-serialization
// computation boundary for independent structural transformation passes.
static llvm::Function* matrix(llvm::Module& M) {
  constexpr auto Name="cpu.f16.mma16816";
  if(auto* Existing=M.getFunction(Name))return Existing;
  auto& C=M.getContext();auto* Ptr=llvm::PointerType::getUnqual(C);
  auto* F=llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(C),{Ptr,Ptr},false),
      llvm::GlobalValue::InternalLinkage,Name,M);
  F->addFnAttr(llvm::Attribute::NoInline);F->addFnAttr(llvm::Attribute::NoUnwind);
  F->addFnAttr(llvm::Attribute::NoSync);F->addFnAttr(llvm::Attribute::WillReturn);
  for(unsigned I=0;I<2;++I){F->addParamAttr(I,llvm::Attribute::NoAlias);F->addParamAttr(I,llvm::Attribute::NoCapture);}
  F->addParamAttr(0,llvm::Attribute::ReadOnly);F->addParamAttr(1,llvm::Attribute::WriteOnly);
  auto* Entry=llvm::BasicBlock::Create(C,"entry",F);llvm::IRBuilder<> B(Entry);
  auto* A=B.CreateAlloca(B.getFloatTy(),B.getInt32(16*16),"a");
  auto* V=B.CreateAlloca(B.getFloatTy(),B.getInt32(16*8),"b");
  auto* D=B.CreateAlloca(B.getFloatTy(),B.getInt32(16*8),"accumulators");
  auto at=[&](llvm::Value* Base,llvm::Value* Row,unsigned Stride,llvm::Value* Col){
    return B.CreateGEP(B.getFloatTy(),Base,B.CreateAdd(B.CreateMul(Row,B.getInt32(Stride)),Col));
  };
  auto read=[&](llvm::Value* Lane,llvm::Value* Field,llvm::Value* High){
    auto* P=B.CreateGEP(B.getInt32Ty(),F->getArg(0),B.CreateAdd(B.CreateMul(Lane,B.getInt32(8)),Field));
    auto* Word=B.CreateLoad(B.getInt32Ty(),P);
    return B.CreateCall(conversion(M,true),{B.CreateLShr(Word,B.CreateMul(High,B.getInt32(16)))});
  };
  loop(B,F,16,"decode.a.row",[&](llvm::Value* I){loop(B,F,16,"decode.a.k",[&](llvm::Value* K){
    auto* Lane=B.CreateAdd(B.CreateMul(B.CreateURem(I,B.getInt32(8)),B.getInt32(4)),B.CreateUDiv(B.CreateURem(K,B.getInt32(8)),B.getInt32(2)));
    auto* Field=B.CreateAdd(B.CreateUDiv(I,B.getInt32(8)),B.CreateMul(B.CreateUDiv(K,B.getInt32(8)),B.getInt32(2)));
    B.CreateStore(read(Lane,Field,B.CreateURem(K,B.getInt32(2))),at(A,I,16,K));
  });});
  loop(B,F,16,"decode.b.k",[&](llvm::Value* K){loop(B,F,8,"decode.b.col",[&](llvm::Value* J){
    auto* Lane=B.CreateAdd(B.CreateMul(J,B.getInt32(4)),B.CreateUDiv(B.CreateURem(K,B.getInt32(8)),B.getInt32(2)));
    auto* Field=B.CreateAdd(B.getInt32(4),B.CreateUDiv(K,B.getInt32(8)));
    B.CreateStore(read(Lane,Field,B.CreateURem(K,B.getInt32(2))),at(V,K,8,J));
  });});
  loop(B,F,16,"decode.c.row",[&](llvm::Value* I){loop(B,F,8,"decode.c.col",[&](llvm::Value* J){
    auto* Lane=B.CreateAdd(B.CreateMul(B.CreateURem(I,B.getInt32(8)),B.getInt32(4)),B.CreateUDiv(J,B.getInt32(2)));
    auto* Field=B.CreateAdd(B.getInt32(6),B.CreateUDiv(I,B.getInt32(8)));
    B.CreateStore(read(Lane,Field,B.CreateURem(J,B.getInt32(2))),at(D,I,8,J));
  });});
  loop(B,F,16,"product.row",[&](llvm::Value* I){loop(B,F,8,"product.col",[&](llvm::Value* J){
    auto* DP=at(D,I,8,J);auto* Initial=B.CreateLoad(B.getFloatTy(),DP);auto* Pre=B.GetInsertBlock();
    auto* Head=llvm::BasicBlock::Create(C,"product.k.head",F);
    auto* Work=llvm::BasicBlock::Create(C,"product.k.body",F);
    auto* Exit=llvm::BasicBlock::Create(C,"product.k.exit",F);
    B.CreateBr(Head);B.SetInsertPoint(Head);
    auto* K=B.CreatePHI(B.getInt32Ty(),2,"k");K->addIncoming(B.getInt32(0),Pre);
    auto* Acc=B.CreatePHI(B.getFloatTy(),2,"sum");Acc->addIncoming(Initial,Pre);
    B.CreateCondBr(B.CreateICmpULT(K,B.getInt32(16)),Work,Exit);B.SetInsertPoint(Work);
    auto* AV=B.CreateLoad(B.getFloatTy(),at(A,I,16,K));auto* BV=B.CreateLoad(B.getFloatTy(),at(V,K,8,J));
    auto* Sum=B.CreateFAdd(Acc,B.CreateFMul(AV,BV));auto* Next=B.CreateAdd(K,B.getInt32(1));B.CreateBr(Head);
    K->addIncoming(Next,Work);Acc->addIncoming(Sum,Work);B.SetInsertPoint(Exit);B.CreateStore(Acc,DP);
  });});
  loop(B,F,32,"encode.lane",[&](llvm::Value* Lane){loop(B,F,2,"encode.row",[&](llvm::Value* Half){
    auto* Row=B.CreateAdd(B.CreateUDiv(Lane,B.getInt32(4)),B.CreateMul(Half,B.getInt32(8)));
    auto* Col=B.CreateMul(B.CreateURem(Lane,B.getInt32(4)),B.getInt32(2));
    auto* Low=pack(B,B.CreateLoad(B.getFloatTy(),at(D,Row,8,Col)));
    auto* High=pack(B,B.CreateLoad(B.getFloatTy(),at(D,Row,8,B.CreateAdd(Col,B.getInt32(1)))));
    B.CreateStore(B.CreateOr(Low,B.CreateShl(High,16)),B.CreateGEP(B.getInt32Ty(),F->getArg(1),B.CreateAdd(B.CreateMul(Lane,B.getInt32(2)),Half)));
  });});
  B.CreateRetVoid();return F;
}

static bool lower(llvm::Module& M, llvm::Function& F, llvm::CallInst* C,
                  unsigned BlockSize) {
  auto* Callee = C->getCalledFunction();
  if (!Callee || Callee->getName() != "hmma16816_f16_warp") return false;
  if (!BlockSize || BlockSize % 32) refuse("F16 matrix collective requires full warps");
  if (C->arg_size() != 9 || !C->getType()->isVoidTy() ||
      !C->getArgOperand(0)->getType()->isPointerTy() ||
      C->getArgOperand(0)->getType()->getPointerAddressSpace() != 0)
    refuse("unsupported F16 matrix collective ABI");
  for (unsigned I = 1; I != 9; ++I)
    if (!C->getArgOperand(I)->getType()->isIntegerTy(32))
      refuse("F16 matrix fragments must be packed i32");

  llvm::IRBuilder<> E(&*F.getEntryBlock().getFirstInsertionPt());
  auto* Scratch = E.CreateAlloca(E.getInt32Ty(), E.getInt32(BlockSize * 8), "cpu.f16.fragments");
  Scratch->setMetadata("cpu.block.shared", llvm::MDNode::get(M.getContext(), {}));
  auto* Results = E.CreateAlloca(E.getInt32Ty(), E.getInt32(BlockSize * 2), "cpu.f16.results");
  Results->setMetadata("cpu.block.shared", llvm::MDNode::get(M.getContext(), {}));
  llvm::IRBuilder<> B(C);
  marker(B, M);
  auto* Index = tid(B, M);
  auto* Local = lane(B, M);
  auto* Base = warpBase(B, M);
  for (unsigned I = 0; I != 8; ++I)
    B.CreateStore(C->getArgOperand(I + 1), B.CreateGEP(B.getInt32Ty(), Scratch,
        B.CreateAdd(B.CreateMul(Index, B.getInt32(8)), B.getInt32(I))));
  marker(B, M);
  auto* Owner=llvm::SplitBlockAndInsertIfThen(B.CreateICmpEQ(Local,B.getInt32(0)),C,false);
  llvm::IRBuilder<> O(Owner);
  O.CreateCall(matrix(M),{
      O.CreateGEP(O.getInt32Ty(),Scratch,O.CreateMul(Base,O.getInt32(8))),
      O.CreateGEP(O.getInt32Ty(),Results,O.CreateMul(Base,O.getInt32(2)))});
  B.SetInsertPoint(C);marker(B,M);
  for(unsigned I=0;I<2;++I) {
    auto* Value=B.CreateLoad(B.getInt32Ty(),B.CreateGEP(B.getInt32Ty(),Results,
        B.CreateAdd(B.CreateMul(tid(B,M),B.getInt32(2)),B.getInt32(I))));
    B.CreateStore(Value,B.CreateGEP(B.getInt32Ty(),C->getArgOperand(0),B.getInt32(I)));
  }
  marker(B, M);
  C->eraseFromParent();
  return true;
}
} // namespace cpu_fp16
