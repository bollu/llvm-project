//===- Dominance.cpp - Dominator analysis for CFGs ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of dominance related classes and instantiations of extern
// templates.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm-c/Object.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"
#include <algorithm>
#include <sys/cdefs.h>

using namespace mlir;
using namespace mlir::detail;

template class llvm::DominatorTreeBase<Block, /*IsPostDom=*/false>;
template class llvm::DominatorTreeBase<Block, /*IsPostDom=*/true>;
template class llvm::DomTreeNodeBase<Block>;

/// Return true if the region with the given index inside the operation
/// has SSA dominance.
static bool hasSSADominance(Operation *op, unsigned index) {
  auto kindInterface = dyn_cast<RegionKindInterface>(op);
  return op->isRegistered() &&
         (!kindInterface || kindInterface.hasSSADominance(index));
}

const int DEBUG = 1;
#define DEBUG_TYPE "dom"

using llvm::dbgs;

//===----------------------------------------------------------------------===//
// DT
//===----------------------------------------------------------------------===//

void DT::debug_print(llvm::raw_ostream &os) const {
  std::set<DTNode *> seen;
  os << "DT {\n";
  for (DTNode *n : this->Nodes) {
    n->print(os);
    os << "\n--\n";
  }
  os << "\n} // end DT\n";
}

//===----------------------------------------------------------------------===//
// DTNode
//===----------------------------------------------------------------------===//

int DTNode::Count = 0;

void DTNode::print(llvm::raw_ostream &os) const {
  switch (this->kind) {
  case Kind::DTBlock:
    os << this << " [ix|" << this->DebugIndex << "] ";
    os << "[bb|" << b << "\n";
    b->print(os);
    os << "]";
    return;
    // case Kind::DTExit:
    //   os << this << " ";
    //   os << "exit-region["
    //      << "\n"
    //      << *r->getParentOp() << "]";
    //   break;

  case Kind::DTOpExit:
    os << this << " [ix|" << this->DebugIndex << "] ";
    os << "[exit-op|\n" << *op << "]";
    return;

  case Kind::DTToplevelEntry:
    os << this << " [ix:|" << this->DebugIndex << "] [entry]";
    return;
  }

  assert(false && "unknown type of node");
  // if (this->successors.size()) { os << "\n"; }
  // for (DTNode *succ : this->successors) {
  //   succ->print(os, indent + 2);
  //   os << "\n";
  // }
}


//===----------------------------------------------------------------------===//
// DominanceInfoBase
//===----------------------------------------------------------------------===//

// bool isRunRegionOp(Operation *op) { return false; }

// void processRegionDom(
//     DT *dt, DenseMap<Region *, std::pair<DTNode *, DTNode *>> &R2EntryExit,
//     DenseMap<mlir::Block *, std::pair<DTNode *, DTNode *>> &Block2Node,
//     DenseMap<Operation *, DTNode *> &Op2Node, Region *r);

// void processOpDom(
//     DT *dt, DenseMap<Region *, std::pair<DTNode *, DTNode *>> &R2EntryExit,
//     DenseMap<mlir::Block *, std::pair<DTNode *, DTNode *>> &Block2Node,
//     DenseMap<Operation *, DTNode *> &Op2Node, Operation *op) {
//   const int numRegions = op->getNumRegions();
//   for (int i = 0; i < numRegions; i++) {
//     Region &R = op->getRegion(i);
//     processRegionDom(dt, R2EntryExit, Block2Node, Op2Node, &R);
//   }
// }

// void processRegionPostDom(
//     DT *dt, DenseMap<Region *, std::pair<DTNode *, DTNode *>> &R2EntryExit,
//     DenseMap<mlir::Block *, std::pair<DTNode *, DTNode *>> &Block2Node,
//     DenseMap<Operation *, DTNode *> &Op2Node, Region *r);

// void processOpPostDom(
//     DT *dt, DenseMap<Region *, std::pair<DTNode *, DTNode *>> &R2EntryExit,
//     DenseMap<mlir::Block *, std::pair<DTNode *, DTNode *>> &Block2Node,
//     DenseMap<Operation *, DTNode *> &Op2Node, Operation *op) {
//   const int numRegions = op->getNumRegions();
//   for (int i = 0; i < numRegions; i++) {
//     Region &R = op->getRegion(i);
//     processRegionPostDom(dt, R2EntryExit, Block2Node, Op2Node, &R);
//   }
// }

void getRegionsFromValue(Value v, std::set<mlir::Region *> &out) {
  assert(!v.dyn_cast<BlockArgument>() &&
         "region value cannot be block argument!");

  // if (RgnValOp val = v.getDefiningOp<RgnValOp>()) {
  //   out.insert(&val.getRegion());
  // } else {
  Operation *op = v.getDefiningOp();
  assert(op && "value that is not a block argument must be an op!");
  for (mlir::Value operand : op->getOperands()) {
    getRegionsFromValue(operand, out);
  }
  // }
}

void processRegionPostDom(
    DT *dt, DenseMap<Region *, std::pair<DTNode *, DTNode *>> &R2EntryExit,
    DenseMap<mlir::Block *, std::pair<DTNode *, DTNode *>> &Block2Node,
    DenseMap<Operation *, DTNode *> &Op2Node, mlir::Region *R,
    DTNode *ParentOpExit) {

  assert(false && "have not thought about this");
  /*
  assert(R->getBlocks().size() > 0);
  // for each block, create entry/exit nodes.
  for (mlir::Block &B : *R) {
    // "entry node" for this block.
    DTNode *BNode = DTNode::newBlock(&B, dt);
    dt->Nodes.push_back(BNode);
    Block2Node[&B].first = Block2Node[&B].second = BNode;
  }

  Block &EntryBlock = R->getBlocks().front();
  DTNode *RegionEntry = Block2Node[&EntryBlock].first;
  assert(ParentOpExit->kind == DTNode::Kind::DTOpExit);
  // dt->Nodes.push_back(RegionExit);

  R2EntryExit[R] = {RegionEntry, ParentOpExit};

  for (mlir::Block &B : *R) {
    for (Operation &Op : B) {
      // recursively process regions.
      processOpPostDom(dt, R2EntryExit, Block2Node, Op2Node, &Op);

      Op2Node[&Op] = Block2Node[&B].second;

      // return like op. exit to region exit.
      if (Op.hasTrait<OpTrait::IsTerminator>() &&
          Op.hasTrait<OpTrait::ReturnLike>()) {
        // add edit to exit block of region
        DTNode *ThisExit = Block2Node[&B].second;
        // RegionExit->addSuccessor(ThisExit);
        // ThisExit->addSuccessor(RegionExit);
        continue;
      }

      // not a return like terminator.
      if (Op.hasTrait<OpTrait::IsTerminator>() &&
          !Op.hasTrait<OpTrait::ReturnLike>()) {
        for (BlockOperand &NextB : Op.getBlockOperands()) {
          if (DEBUG) {
            llvm::dbgs() << "creating next block links for |" << Op
                         << "| to: " << NextB.get() << "\n";
            getchar();
          }
          DTNode *ThisExit = Block2Node[&B].second;
          DTNode *NextEntry = Block2Node[NextB.get()].first;
          NextEntry->addSuccessor(ThisExit);
          // ThisExit->addSuccessor(NextEntry);
        }
        continue;
      }
    }
  }
  */
}

void processRegionDom(
    DT *dt, DenseMap<Region *, std::pair<DTNode *, DTNode *>> &R2EntryExit,
    DenseMap<mlir::Block *, std::pair<DTNode *, DTNode *>> &Block2EntryExit,
    DenseMap<Operation *, DTNode *> &Op2Dominator, DTNode *ParentOpDominator,
    DTNode *ParentOpExit, mlir::Region *R) {

  if (R->getBlocks().size() == 0) {
    return;
  }

  assert(R->getBlocks().size() > 0);
  // for each block, create entry/exit nodes.
  for (mlir::Block &B : *R) {
    // "entry node" for this block.
    DTNode *BNode = DTNode::newBlock(&B, dt);
    Block2EntryExit[&B].first = Block2EntryExit[&B].second = BNode;
  }

  Block &EntryBlock = R->getBlocks().front();
  DTNode *RegionEntryNode = Block2EntryExit[&EntryBlock].first;
  assert(RegionEntryNode);

  R2EntryExit[R] = {RegionEntryNode, ParentOpExit};

  // Hack : make dominance lexically scoped for entry.
  ParentOpDominator->addSuccessor(RegionEntryNode);

  // Step 1: build data structures
  for (mlir::Block &B : *R) {
    for (Operation &Op : B) {
      // Step 1a: ecursively process regions.
      const int numRegions = Op.getNumRegions();
      DTNode *OpParent = Block2EntryExit[&B].second;
      Op2Dominator[&Op] = OpParent;
      if (numRegions > 0) {
        DTNode *OpExit = DTNode::newOpExit(&Op, dt);
        for (int i = 0; i < numRegions; i++) {
          Region &R = Op.getRegion(i);
          processRegionDom(dt, R2EntryExit, Block2EntryExit, Op2Dominator,
                           OpParent, OpExit, &R);
        }
        // current final dominating thing is OpExit
        Block2EntryExit[&B].second = OpExit;
      }

      // return like op. exit to region exit.
      if (Op.hasTrait<OpTrait::IsTerminator>() &&
          Op.hasTrait<OpTrait::ReturnLike>()) {
        // add edit to exit block of region
        DTNode *ThisExit = Block2EntryExit[&B].second;
        // Hack : make dominance lexically scoped for exit.
        ThisExit->addSuccessor(ParentOpExit);
      }

      // not a return like terminator.
      else if (Op.hasTrait<OpTrait::IsTerminator>() &&
               !Op.hasTrait<OpTrait::ReturnLike>()) {
        for (BlockOperand &NextB : Op.getBlockOperands()) {
          if (DEBUG) {
            llvm::dbgs() << "creating next block links for |" << Op
                         << "| to: " << NextB.get() << "\n";
            getchar();
          }
          DTNode *ThisExit = Block2EntryExit[&B].second;
          DTNode *NextEntry = Block2EntryExit[NextB.get()].first;
          ThisExit->addSuccessor(NextEntry);
        }
      }
    }
  }
}

template <bool IsPostDom>
void DominanceInfoBase<IsPostDom>::recalculate(Operation *op) {

  // ModuleOp module = dyn_cast<ModuleOp>(op);
  assert(isa<ModuleOp>(op) || isa<FuncOp>(op));

  this->R2EntryExit.clear();
  this->Block2EntryExit.clear();
  this->Op2Node.clear();

  // op->walk([&](mlir::FuncOp f) {
  DT *dt = new DT();
  auto dtBase = std::make_unique<DTBaseT>();

  if (!IsPostDom) {
    DTNode *toplevelEntry = DTNode::newToplevelEntry(op, dt);
    Op2Node[op] = toplevelEntry;
    dt->entry = toplevelEntry;

    DTNode *toplevelExit = DTNode::newOpExit(op, dt);
    for (int i = 0; i < op->getNumRegions(); ++i) {
      Region &r = op->getRegion(i);
      processRegionDom(dt, this->R2EntryExit, this->Block2EntryExit,
                       this->Op2Node, toplevelEntry, toplevelExit, &r);
    }

    llvm::errs() << "trying recalculate...\n";
    dtBase->recalculate(*dt);
    llvm::errs() << "\nSUCCESS!\n";
    this->tree = dtBase.get();
    llvm::errs() << "\nRECALCULATED tree: |" << this->tree << "|\n";
    getchar();

    // if (DEBUG) {
    //   llvm::dbgs() << "\n\n@@@@processing function.. |" << f << "\n";
    // }
  } else {
    assert(false && "unimplemented");
    // processOpPostDom(dt, this->R2EntryExit, this->Block2EntryExit,
    // this->Op2Node,
    //              f);
    // dt->entry = this->R2EntryExit[&f.getRegion()].second;
    // dominanceInfo->recalculate(*dt);
    // if (DEBUG) {
    //   llvm::dbgs() << "\n\n@@@@processing function.. |" << f << "\n";
    // }
  }

  // for (int i = 0; i < dt->Nodes.size(); ++i) {
  //   // dt->Nodes[i]->Info = dominanceInfo.get();
  //   dt->Nodes[i]->DebugIndex = i;
  // }

  if (DEBUG) {

    for (int i = 0; i < dt->Nodes.size(); ++i) {
      for (int j = 0; j < dt->Nodes.size(); ++j) {
        llvm::dbgs() << "dominates(" << i << " " << j
                     << ", isPostDom:" << IsPostDom << ") = "
                     << dtBase->dominates(dt->Nodes[i], dt->Nodes[j])
                     << "\n";
      }
    }

    for (int i = 0; i < dt->Nodes.size(); ++i) {
      llvm::dbgs() << "\n##" << (dt->entry == dt->Nodes[i] ? " ENTRY" : "")
                   << ":\n";
      dt->Nodes[i]->print(llvm::dbgs());
      llvm::dbgs() << "\n==\n";
    }

    llvm::dbgs() << "edges:\n";
    for (int i = 0; i < dt->Nodes.size(); ++i) {
      for (int j = 0; j < dt->Nodes[i]->successors.size(); ++j) {
        llvm::dbgs() << dt->Nodes[i]->DebugIndex << " -> " << dt->Nodes[i]->successors[j]->DebugIndex
                     << "\n";
      }
    }

    int FD;
    llvm::sys::fs::openFileForWrite(
        "/home/bollu/temp/graph.dot", FD);
    llvm::raw_fd_ostream O(FD, /*shouldClose=*/true);
    llvm::WriteGraph(O, dt);
  }
}

/// Walks up the list of containers of the given block and calls the
/// user-defined traversal function for every pair of a region and block that
/// could be found during traversal. If the user-defined function returns true
/// for a given pair, traverseAncestors will return the current block. Nullptr
/// otherwise.
template <typename FuncT>
Block *traverseAncestors(Block *block, const FuncT &func) {
  assert(false);
  // assert(false && "unimplemented");
  // // Invoke the user-defined traversal function in the beginning for the current
  // // block.
  // if (func(block))
  //   return block;

  // Region *region = block->getParent();
  // while (region) {
  //   Operation *ancestor = region->getParentOp();
  //   // If we have reached to top... return.
  //   if (!ancestor || !(block = ancestor->getBlock()))
  //     break;

  //   // Update the nested region using the new ancestor block.
  //   region = block->getParent();

  //   // Invoke the user-defined traversal function and check whether we can
  //   // already return.
  //   if (func(block))
  //     return block;
  // }
  // return nullptr;
}


template <bool IsPostDom>
Block *
DominanceInfoBase<IsPostDom>::findNearestCommonDominator(Block *a,
                                                         Block *b) const {
    DTNode *na = [&]() -> DTNode* {
      auto it = this->Block2EntryExit.find(a);
      if (it == this->Block2EntryExit.end()) {
        return nullptr;
      }  else {
        return it->second.first;
      }
    }();

    DTNode *nb = [&]() -> DTNode* {
      auto it = this->Block2EntryExit.find(b);
      if (it == this->Block2EntryExit.end()) {
        return nullptr;
      }  else {
        return it->second.first;
      }
    }();

    if (!na || !nb) { return nullptr; }

    llvm::errs() << "findNearestCommonDominator(na:" << *na << " " <<
      na->DebugIndex << ", nb:" << *nb << " " << nb->DebugIndex << ", tree: " << this->tree << ")\n";
    assert(na); assert(nb);

    DTNode *nearest = tree->findNearestCommonDominator(na, nb);
    assert(nearest->kind != DTNode::Kind::DTToplevelEntry);
    if (nearest->kind == DTNode::Kind::DTBlock) {
      return nearest->getBlock();
    } else {
      return nearest->getOp()->getBlock();
    }
}

template <bool IsPostDom>
DominanceInfoNode *DominanceInfoBase<IsPostDom>::getNode(Block *a) {
  assert(false && "unimplemented");
}

/// Return true if the specified block A properly dominates block B.
template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::properlyDominates(Block *a, Block *b) const {
  assert(false && "unimplemented");
  
}

/// Return true if the specified block is reachable from the entry block of its
/// region.
template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::isReachableFromEntry(Block *a) const {
  if (!IsPostDom) {
    DTNode *aNode = [&]() {
      auto it = this->Block2EntryExit.find(a);
      if (it != this->Block2EntryExit.end()) {
        return it->second.first;
      } else {
        return (DTNode *)nullptr;
      }
    }();

    if (aNode) {
      return this->tree->isReachableFromEntry(aNode);
    } else {
      return false;
    }
  } else {

  }

}

template class detail::DominanceInfoBase</*IsPostDom=*/true>;
template class detail::DominanceInfoBase</*IsPostDom=*/false>;

//===----------------------------------------------------------------------===//
// DominanceInfo
//===----------------------------------------------------------------------===//


bool properlyDominatesReal(llvm::DominatorTreeBase<DTNode, false> *tree, DTNode *a, DTNode *b) {
  assert(tree && "expected legal tree");
  if (a == nullptr || b == nullptr) { return false; }

  if (a == b) { return false; }

  if (a->kind == DTNode::Kind::DTToplevelEntry) { return true; }
  else if (a->kind == DTNode::Kind::DTBlock) {
    // if `b` is a block, then it's not `a` (since we've already checked). So dom => properly dom.
    // if `b` is an op, then the DTNode could be `a` (since it's an op), but a block dominantes all ops in it, so properly d
    return tree->dominates(a, b);
  }
  else {
    // we setup dominance correctly
    assert(a->kind == DTNode::Kind::DTOpExit);
    return tree->properlyDominates(a, b);
  }
}

bool DominanceInfo::properlyDominates(Block *a, Block *b) const {
    assert(false && "unimplemented");

  return false;
}


/// Return true if operation A properly dominates operation B.
bool DominanceInfo::properlyDominates(Operation *a, Operation *b) const {
  assert(false && "unimplemented");

  Block *aBlock = a->getBlock(), *bBlock = b->getBlock();
  Region *aRegion = a->getParentRegion();
  unsigned aRegionNum = aRegion->getRegionNumber();
  Operation *ancestor = aRegion->getParentOp();

  // If a or b are not within a block, then a does not dominate b.
  if (!aBlock || !bBlock)
    return false;

  if (aBlock == bBlock) {
    // Dominance changes based on the region type. In a region with SSA
    // dominance, uses inside the same block must follow defs. In other
    // regions kinds, uses and defs can come in any order inside a block.
    if (hasSSADominance(ancestor, aRegionNum)) {
      // If the blocks are the same, then check if b is before a in the block.
      return a->isBeforeInBlock(b);
    }
    return true;
  }

  // Traverse up b's hierarchy to check if b's block is contained in a's.
  if (auto *bAncestor = aBlock->findAncestorOpInBlock(*b)) {
    // Since we already know that aBlock != bBlock, here bAncestor != b.
    // a and bAncestor are in the same block; check if 'a' dominates
    // bAncestor.
    return dominates(a, bAncestor);
  }

  // If the blocks are different, check if a's block dominates b's.
  return properlyDominates(aBlock, bBlock);
}

/// Return true if value A properly dominates operation B.
bool DominanceInfo::properlyDominates(Value a, Operation *b) const {
  DTNode *aNode = nullptr;
  if (Operation *aOp = a.getDefiningOp()) {
    auto ita = this->Op2Node.find(aOp);
    assert(ita != Op2Node.end());
    aNode = ita->second;
  } else {
      // can add new BBs.
      assert(a.isa<BlockArgument>() && "value must be op or block argument");
      BlockArgument arg = a.cast<BlockArgument>();
      auto ita = this->Block2EntryExit.find(arg.getOwner());
      if (ita != this->Block2EntryExit.end()) {
        // assert(ita != this->Block2EntryExit.end());
        aNode = ita->second.first;
      }
  }
  DTNode *bNode = [&]() {
    auto itb = this->Op2Node.find(b);
    assert(itb != this->Op2Node.end());
    return itb->second;
  }();


  assert(false && "unimplemented");
  return properlyDominatesReal(this->tree, aNode, bNode);
  // if (Operation *aOp = a.getDefiningOp()) {
  //   auto ita = this->Op2Node.find(aOp);
  //   auto itb = this->Op2Node.find(b);

  //   // is this a correct over-approximation?
  //   if (ita == Op2Node.end() || itb == Op2Node.end()) {
  //     return false;
  //   }

  //   assert(ita != Op2Node.end());
  //   assert(itb != Op2Node.end());

  //   assert(ita->second->Info == itb->second->Info &&
  //          "both must have same dom info data structure");
  //   DTBaseT *dominanceInfo = (DTBaseT *)ita->second->Info;

  //   if (ita->second == itb->second) {
  //     // these are next to each other in the same BB / are not interleaved with
  //     // a region instruction.
  //     return aOp != b && aOp->isBeforeInBlock(b);
  //   }

  //   return aOp != b && dominanceInfo->dominates(ita->second, itb->second);
  // }

  // assert(a.isa<BlockArgument>() && "value must be op or block argument");
  // // block arguments properly dominate all operations in their own block, so
  // // we use a dominates check here, not a properlyDominates check.
  // return dominates(a.cast<BlockArgument>().getOwner(), b->getBlock());

  // assert(false && "unimplemented");

  // if (auto *aOp = a.getDefiningOp()) {
  //   // Dominance changes based on the region type.
  //   auto *aRegion = aOp->getParentRegion();
  //   unsigned aRegionNum = aRegion->getRegionNumber();
  //   Operation *ancestor = aRegion->getParentOp();
  //   // Dominance changes based on the region type. In a region with SSA
  //   // dominance, values defined by an operation cannot be used by the
  //   // operation. In other regions kinds they can be used the operation.
  //   if (hasSSADominance(ancestor, aRegionNum)) {
  //     // The values defined by an operation do *not* dominate any nested
  //     // operations.
  //     if (aOp->getParentRegion() != b->getParentRegion() && aOp->isAncestor(b))
  //       return false;
  //   }
  //   return properlyDominates(aOp, b);
  // }

  // // block arguments properly dominate all operations in their own block, so
  // // we use a dominates check here, not a properlyDominates check.
  // return dominates(a.cast<BlockArgument>().getOwner(), b->getBlock());
}

void DominanceInfo::updateDFSNumbers() {
  assert(false && "unimplemented");

  for (auto &iter : dominanceInfos)
    iter.second->updateDFSNumbers();
}

//===----------------------------------------------------------------------===//
// PostDominanceInfo
//===----------------------------------------------------------------------===//

/// Returns true if statement 'a' properly postdominates statement b.
bool PostDominanceInfo::properlyPostDominates(Operation *a, Operation *b) {
  assert(false && "unimplemented");

  auto *aBlock = a->getBlock(), *bBlock = b->getBlock();
  auto *aRegion = a->getParentRegion();
  unsigned aRegionNum = aRegion->getRegionNumber();
  Operation *ancestor = aRegion->getParentOp();

  // If a or b are not within a block, then a does not post dominate b.
  if (!aBlock || !bBlock)
    return false;

  // If the blocks are the same, check if b is before a in the block.
  if (aBlock == bBlock) {
    // Dominance changes based on the region type.
    if (hasSSADominance(ancestor, aRegionNum)) {
      // If the blocks are the same, then check if b is before a in the block.
      return b->isBeforeInBlock(a);
    }
    return true;
  }

  // Traverse up b's hierarchy to check if b's block is contained in a's.
  if (auto *bAncestor = a->getBlock()->findAncestorOpInBlock(*b))
    // Since we already know that aBlock != bBlock, here bAncestor != b.
    // a and bAncestor are in the same block; check if 'a' postdominates
    // bAncestor.
    return postDominates(a, bAncestor);

  // If the blocks are different, check if a's block post dominates b's.
  return properlyDominates(aBlock, bBlock);
}
