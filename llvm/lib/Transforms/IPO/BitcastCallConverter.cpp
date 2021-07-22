//===- InlineAlways.cpp - Code to inline always_inline functions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a custom inliner that handles only functions that
// are marked as "always inline".
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/BitcastCallConverter.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/MCA/Instruction.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/X86DisassemblerDecoderCommon.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "inline"

bool isCastable(llvm::Type *SrcTy, llvm::Type *DestTy) {
  if (SrcTy->isVoidTy() && DestTy->isVoidTy()) { return true; }
  if (SrcTy->isPointerTy() && DestTy->isPointerTy()) {
    return true;
  }
  if (SrcTy->isIntegerTy() && DestTy->isIntegerTy()) {
    return true;
  }
  return false;
}

Value *castValue(llvm::IRBuilder<> &Builder, llvm::Value *SrcV,
                 llvm::Type *DestTy) {
  assert(isCastable(SrcV->getType(), DestTy));
  if (SrcV->getType()->isIntegerTy()) {
    const bool IsSigned = false;
    return Builder.CreateIntCast(SrcV, DestTy, IsSigned);
  }
  assert(SrcV->getType()->isPointerTy());
  return Builder.CreatePointerCast(SrcV, DestTy);
}

bool isFnTypeCastable(llvm::FunctionType *src, llvm::FunctionType *dest) {
  if (!isCastable(src->getReturnType(), dest->getReturnType())) {
    return false;
  }
  if (src->getNumParams() != dest->getNumParams()) {
    return false;
  }
  for (int i = 0; i < (int)src->getNumParams(); ++i) {
    if (!isCastable(src->getParamType(i), dest->getParamType(i))) {
      return false;
    }
  }
  return true;
}

FunctionType *getFnFromFnPtrTy(Type *t) {
  assert(t && "expected legal type");
  PointerType *pty = dyn_cast<PointerType>(t);
  if (!pty) {
    return nullptr;
  }
  return dyn_cast<FunctionType>(pty->getPointerElementType());
}
PreservedAnalyses BitcastCallConverterPass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  // llvm::SmallVector<Instruction*, 4> CallWithCast;
  SmallVector<Instruction *, 4> CallsToErase;
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        CallInst *C = dyn_cast<CallInst>(&I);
        if (!C) {
          continue;
        }
        BitCastOperator *Cast =
            dyn_cast<BitCastOperator>(C->getCalledOperand());
        if (!Cast) {
          continue;
        }

        Function *CalledF = dyn_cast<Function>(Cast->getOperand(0));
        if (!CalledF) {
          continue;
        }


        llvm::errs() << "call with cast to fn |" << CalledF->getName() << "|\n";
        llvm::errs() << "\tcast: " << *C << "|\n";
        llvm::errs() << "\tcast src type: " << *Cast->getSrcTy() << "\n";
        ;
        llvm::errs() << "\tcast dst type: " << *Cast->getDestTy() << "\n";
        llvm::errs() << "\tcast src is ptr: " << Cast->getSrcTy()->isPointerTy()
                     << "\n";
        llvm::errs() << "\tcast src is fn: " << Cast->getSrcTy()->isFunctionTy()
                     << "\n";

        FunctionType *FSrcTy = getFnFromFnPtrTy(Cast->getSrcTy());
        FunctionType *FDstTy = getFnFromFnPtrTy(Cast->getDestTy());

        assert(FSrcTy && FDstTy);
        assert(CalledF->getName() == "main_lean_custom_entrypoint_hack" || 
            isFnTypeCastable(FSrcTy, FDstTy));
        if (!FSrcTy || !FDstTy) {
          continue;
        }
        if (!isFnTypeCastable(FSrcTy, FDstTy)) {
          continue;
        }
        // getchar();


        // TODO: add a check that is possible.
        llvm::IRBuilder<> Builder(&BB);
        Builder.SetInsertPoint(C);
        llvm::SmallVector<Value *, 4> CallArgs;

        for (int i = 0; i < C->arg_size(); ++i) {
          Value *V =
              castValue(Builder, C->getArgOperand(i), FSrcTy->getParamType(i));
          // Value *V = Builder.CreateBitOrPointerCast(U.get(),
          // FSrcTy->getParamType(ix));
          CallArgs.push_back(V);
        }


        CallInst *NewCall = Builder.CreateCall(CalledF, CallArgs);
        llvm::errs() << "\tNewCall: " << *NewCall << "\n";
        // assert(CalledF->getName() != "lean_dec");

        // v super hack! Force all such calls to be inlined...

        AttrBuilder AB;
        AB.addAttribute(Attribute::AlwaysInline);
        CalledF->addAttributes(AttributeList::FunctionIndex, AB);
        // CalledF->addFnAttr(Attribute::AlwaysInline); 
        if (!CalledF->getReturnType()->isVoidTy()) {
            Value *VCasted = castValue(Builder, NewCall, FDstTy->getReturnType());
            // Value *VCasted = Builder.CreateBitOrPointerCast(VCallRet,
            // FDstTy->getReturnType());
            C->replaceAllUsesWith(VCasted);
        }
        CallsToErase.push_back(C); // can't invalidate iterator.

      }
    }
  }

  for (Instruction *I : CallsToErase) {
    assert(I->getNumUses() == 0);
    I->eraseFromParent();
  }

  return PreservedAnalyses::none();
}
