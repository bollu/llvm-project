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
#include "mlir/Analysis/NestedMatcher.h"
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
#include "llvm/CodeGen/ISDOpcodes.h"
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


const int DEBUG = 0;
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

template <bool IsPostDom>
void addSuccessor(DTNode *parent, DTNode *child) {
  if (IsPostDom) {
    child->addSuccessor(parent);
  } else {
    parent->addSuccessor(child);
  }
}

template<bool IsPostDom>
void processRegionDom(
    DT *dt, DenseMap<Region *, std::pair<DTNode *, DTNode *>> &R2EntryExit,
    DenseMap<mlir::Block *, std::pair<DTNode *, DTNode *>> &Block2EntryExit,
    DenseMap<Operation *, DTNode *> &Op2Dominator, DTNode *ParentOpEntry,
    DTNode *ParentOpExit, 
    Region *root, 
    DenseMap<Region *, Region *> Region2Root,
    DenseMap<Region *, std::pair<DT*,  typename DominanceInfoBase<IsPostDom>::DTBaseT *> > &Region2Tree,
    mlir::Region *R) {

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
  DTNode *RegionEntry = Block2EntryExit[&EntryBlock].first;
  assert(RegionEntry);

  R2EntryExit[R] = {RegionEntry, ParentOpExit};

  // Hack : make dominance lexically scoped for entry.
  if (ParentOpEntry->kind == DTNode::Kind::DTToplevelEntry ||
     R->getParentOp()->mightHaveTrait<OpTrait::IsIsolatedFromAbove>()) {
    // DAGRoots.insert(RegionEntry);
    dt = new DT();
    dt->entry = RegionEntry;
    root = R;
    Region2Root[R] = root;
    Region2Tree[R] = {dt, nullptr}; // as a marker for a domtree that needs to be recalculated.
    llvm::errs() << "\n===\nadded DAG root: ";
    llvm::errs() << *R->getParentOp();
    llvm::errs() << "\n===\n";
    getchar();

  } else {
    Region2Root[R] = root;
    ParentOpEntry->addSuccessor(RegionEntry);
  }

  // Step 1: build data structures
  for (mlir::Block &B : *R) {
    for (Operation &Op : B) {
      // Step 1a: ecursively process regions.
      const int numRegions = Op.getNumRegions();
      DTNode *OpEntry = Block2EntryExit[&B].second;
      DTNode *OpExit = DTNode::newOpExit(&Op, dt);
      Op2Dominator[&Op] = OpExit;
      if (numRegions > 0) {
        for (int i = 0; i < numRegions; i++) {
            Region &R = Op.getRegion(i);
            processRegionDom<IsPostDom>(dt, R2EntryExit, Block2EntryExit, Op2Dominator,
                             OpEntry, OpExit, root, Region2Root, Region2Tree, &R);
        }
      } else {
        OpEntry->addSuccessor(OpExit);
      }
      // current final dominating thing is OpExit
      Block2EntryExit[&B].second = OpExit;


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
  // assert(isa<ModuleOp>(op) || isa<FuncOp>(op));

  this->R2EntryExit.clear();
  this->Block2EntryExit.clear();
  this->Op2Node.clear();

  // op->walk([&](mlir::FuncOp f) {
  DT *dt = new DT();
  DTNode *toplevelEntry = DTNode::newToplevelEntry(op, dt);
  Op2Node[op] = toplevelEntry;
  dt->entry = toplevelEntry;

  DTNode *toplevelExit = DTNode::newOpExit(op, dt);
  for (int i = 0; i < op->getNumRegions(); ++i) {
    Region &r = op->getRegion(i);
    processRegionDom<IsPostDom>(dt, this->R2EntryExit, this->Block2EntryExit,
                     this->Op2Node, toplevelEntry, toplevelExit, &r, this->Region2Root, this->Region2Tree, &r);
  }

  if (DEBUG) {
    for (int i = 0; i < dt->Nodes.size(); ++i) {
      llvm::dbgs() << "\n--" << (dt->entry == dt->Nodes[i] ? " ENTRY" : "")
                   << "--\n";
      dt->Nodes[i]->print(llvm::dbgs());
      llvm::dbgs() << "\n==\n";
    }

    llvm::dbgs() << "edges:\n";
    for (int i = 0; i < dt->Nodes.size(); ++i) {
      for (int j = 0; j < dt->Nodes[i]->successors.size(); ++j) {
        llvm::dbgs() << dt->Nodes[i]->DebugIndex << " -> "
                     << dt->Nodes[i]->successors[j]->DebugIndex << "\n";
      }
    }

    int FD;
    llvm::sys::fs::openFileForWrite("/home/bollu/temp/graph.dot", FD);
    llvm::raw_fd_ostream O(FD, /*shouldClose=*/true);
    llvm::WriteGraph(O, dt, /*shortNames=*/ false, op->getName().getStringRef());
  }

  llvm::errs() << "done with recursion!\n";
  getchar();

  for(auto it : Region2Tree) {
    it.second.second = new DTBaseT();
    llvm::errs() << "asking recalculate!\n";
    this->R2EntryExit[it.first].first->print(llvm::errs());
    it.second.second->recalculate(*it.second.first);
  }

  llvm::errs() << "built all domtrees!\n";
  getchar();

  // this->tree = new DTBaseT(); // std::make_unique<DTBaseT>();
  // tree->recalculate(*dt);
  // if (DEBUG) {

  //   llvm::dbgs() << "\nRECALCULATED tree: |" << "?????" << "|\n";

  //   for (int i = 0; i < dt->Nodes.size(); ++i) {
  //     for (int j = 0; j < dt->Nodes.size(); ++j) {
  //       llvm::dbgs() << "properlyDominates(" << dt->Nodes[i]->DebugIndex << " "
  //                    << dt->Nodes[j]->DebugIndex << ", isPostDom:" << IsPostDom
  //                    << ") = " << "????" // tree->dominates(dt->Nodes[i], dt->Nodes[j])
  //                    << "\n";
  //     }
  //   }
  //   getchar();
  // }
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
  // // Invoke the user-defined traversal function in the beginning for the
  // current
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
  DTNode *na = [&]() -> DTNode * {
    auto it = this->Block2EntryExit.find(a);
    if (it == this->Block2EntryExit.end()) {
      return nullptr;
    } else {
      return IsPostDom ? it->second.second : it->second.first;
    }
  }();

  DTNode *nb = [&]() -> DTNode * {
    auto it = this->Block2EntryExit.find(b);
    if (it == this->Block2EntryExit.end()) {
      return nullptr;
    } else {
      return IsPostDom ? it->second.second : it->second.first;
    }
  }();

  if (!na || !nb) { return nullptr; }
  assert(na);
  assert(nb);

  // DTNode *nearest = tree->findNearestCommonDominator(na, nb);
  DTNode *nearest = nullptr; // tree->findNearestCommonDominator(na, nb);
  assert(nearest->kind != DTNode::Kind::DTToplevelEntry);
  if (nearest->kind == DTNode::Kind::DTBlock) {
    return nearest->getBlock();
  } else {
    assert(nearest->kind == DTNode::Kind::DTOpExit);
    // llvm::DomTreeNodeBase<DTNode> *nearestBase = tree->getNode(nearest);
    llvm::DomTreeNodeBase<DTNode> *nearestBase = nullptr;
    return nearest->getOp()->getBlock();
  }
}

template <bool IsPostDom>
DominanceInfoNode *DominanceInfoBase<IsPostDom>::getNode(Block *a) {
  auto it = this->Block2EntryExit.find(a);
  assert(it != this->Block2EntryExit.end());
  // if post dom return exit else entry
  assert(false && "unimplemented");

  // return tree->getNode(IsPostDom ? it->second.second : it->second.first);
}

/// Return true if the specified block A properly dominates block B.
template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::properlyDominates(Block *a, Block *b) const {
  if (a == nullptr || b == nullptr) {
    return false;
  }
  DTNode *anode = [&]() -> DTNode * {
    auto it = this->Block2EntryExit.find(a);
    if (it == Block2EntryExit.end()) {
      return nullptr;
    }
    return IsPostDom ? it->second.second : it->second.first;
  }();

  DTNode *bnode = [&]() -> DTNode * {
    auto it = this->Block2EntryExit.find(b);
    if (it == Block2EntryExit.end()) {
      return nullptr;
    }
    return IsPostDom ? it->second.second : it->second.first;
  }();

  if (!anode || !bnode) {
    return false;
  }
  assert(false && "unimplemented");

  // return tree->properlyDominates(anode, bnode);
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
      // assert(false && "unimplemented");
      auto it = this->Region2Root.find(a->getParent());
      if (it == Region2Root.end()) { return false; }

      assert(it != this->Region2Root.end());
      Region *root = it->second;
      auto itTree = this->Region2Tree.find(root);
      assert(itTree != this->Region2Tree.end()); 
      return itTree->second.second->isReachableFromEntry(aNode);
      // return this->tree->isReachableFromEntry(aNode);
    } else {
      return false;
    }
  } else {
    assert(false && "unimplemented");
  }
}

template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::isReachableFromParentRegion(Block *a) const {
  Region *r = a->getParent();
  auto rit = this->R2EntryExit.find(r);
  if (rit == this->R2EntryExit.end()) {
    return false;
  }

  DTNode *parentNode = IsPostDom ? rit->second.second : rit->second.first;

  DTNode *aNode = [&]() {
    auto it = this->Block2EntryExit.find(a);
    if (it != this->Block2EntryExit.end()) {
      return IsPostDom ? it->second.second : it->second.first;
    } else {
      return (DTNode *)nullptr;
    }
  }();

  // does this actually return whether you're reachable? I am not sure.
    if (parentNode->getParent() != aNode->getParent()) { return false; }
    else { assert(false); }

  // return tree->dominates(parentNode, aNode);
}

template class detail::DominanceInfoBase</*IsPostDom=*/true>;
template class detail::DominanceInfoBase</*IsPostDom=*/false>;


/// Return true if the region with the given index inside the operation
/// has SSA dominance.
template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::hasSSADominance(Operation *op, unsigned index) {
  auto kindInterface = dyn_cast<RegionKindInterface>(op);
  return op->isRegistered() &&
         (!kindInterface || kindInterface.hasSSADominance(index));
}

template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::hasSSADominance(Region &r) {
  Operation *op = r.getParentOp();
  const int index =  r.getRegionNumber();
  if (!op) { return false; }
  return hasSSADominance(op, index);
}


//===----------------------------------------------------------------------===//
// DominanceInfo
//===----------------------------------------------------------------------===//

/// Return true if operation A properly dominates operation B.
bool DominanceInfo::properlyDominates(Operation *a, Operation *b) const {
  if (!a || !b) {
    return false;
  }

  DTNode *anode = nullptr;
  {
    auto it = this->Op2Node.find(a);
    if (it == Op2Node.end()) { return false; }
    anode = it->second;
  }

  DTNode *bnode = nullptr; 
  {
    auto it = this->Op2Node.find(b);
    if (it == Op2Node.end()) { return false; }
    bnode = it->second;
  }

  assert(anode);
  assert(bnode);
  assert (anode->kind == DTNode::Kind::DTOpExit || anode->kind == DTNode::Kind::DTToplevelEntry);
  assert (bnode->kind == DTNode::Kind::DTOpExit || bnode->kind == DTNode::Kind::DTToplevelEntry);
 if (anode->getParent() != bnode->getParent()) { return false; }
  else { assert(false); }

  // return tree->properlyDominates(anode, bnode);
}

/// Return true if value A properly dominates operation B.
bool DominanceInfo::properlyDominates(Value a, Operation *b) const {
  auto bRoot = this->Region2Root.find(b->getParentRegion());
  if (bRoot == this->Region2Root.end()) { return false; }
  auto bTree = this->Region2Tree.find(bRoot->second); 
  if(bTree == this->Region2Tree.end()) { return false; }

  DTNode *bNode = [&]() {
    auto itb = this->Op2Node.find(b);
    assert(itb != this->Op2Node.end());
    return itb->second;
  }();

  assert(bNode->kind == DTNode::Kind::DTOpExit);
  
  if (Operation *aOp = a.getDefiningOp()) {
    auto aRoot = this->Region2Root.find(aOp->getParentRegion());
    if (aRoot == this->Region2Root.end()) { return false; }

    if (aRoot->second != bRoot->second) { return false; }

    auto ita = this->Op2Node.find(aOp);
    assert(ita != Op2Node.end());

    DTNode *aNode = ita->second;
    assert(aNode->kind == DTNode::Kind::DTOpExit);
    if (aNode->getParent() != bNode->getParent()) { return false; }
    else { 
      return bTree->second.second->properlyDominates(aNode, bNode);
      // return tree->properlyDominates(aNode, bNode);

    }   
  } else {
    assert(a.isa<BlockArgument>() && "value must be op or block argument");
    BlockArgument arg = a.cast<BlockArgument>();
    auto aRoot = this->Region2Root.find(arg.getParentRegion());
    if (aRoot == this->Region2Root.end()) { return false; }
    if (aRoot->second != bRoot->second) { return false; }


    auto ita = this->Block2EntryExit.find(arg.getOwner());
    if (ita == this->Block2EntryExit.end()) {
      return false;
    }
    // argument block properly dominates operation if operation is in the block.
    DTNode *aNode = ita->second.first;
  if (aNode->getParent() != bNode->getParent()) { return false; }
    else { assert(false); }

    // return tree->properlyDominates(aNode, bNode);
  }
}

void DominanceInfo::updateDFSNumbers() { assert(false && "unimplemented"); }

//===----------------------------------------------------------------------===//
// PostDominanceInfo
//===----------------------------------------------------------------------===//

/// Returns true if statement 'a' properly postdominates statement b.
bool PostDominanceInfo::properlyPostDominates(Operation *a, Operation *b) {
  if (!a || !b) {
    return false;
  }
  DTNode *anode = [&]() -> DTNode * {
    auto it = this->Op2Node.find(a);
    if (it == Op2Node.end()) {
      return nullptr;
    }
    return it->second;
  }();

  DTNode *bnode = [&]() -> DTNode * {
    auto it = this->Op2Node.find(b);
    if (it == Op2Node.end()) {
      return nullptr;
    }
    return it->second;
  }();

  if (!anode || !bnode) {
    return false;
  }
  assert(false);
  // return tree->properlyDominates(anode, bnode);
}


