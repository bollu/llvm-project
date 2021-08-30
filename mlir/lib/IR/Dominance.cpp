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
#include "mlir/Dialect/Rgn/RgnDialect.h"
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
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
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

void DTNode::print(llvm::raw_ostream &os) {
  switch (this->kind) {
  case Kind::DTBlock:
    os << this << " ";
    os << "bb[" << b << "\n";
    b->print(os);
    os << "]";
    break;
  case Kind::DTExit:
    os << this << " ";
    os << "exit-region["
       << "\n"
       << *r->getParentOp() << "]";
    break;

  case Kind::DTOp:
    os << this << " ";
    os << "op[\n" << *op << "]";
    break;
  }

  // if (this->successors.size()) { os << "\n"; }
  // for (DTNode *succ : this->successors) {
  //   succ->print(os, indent + 2);
  //   os << "\n";
  // }
}

//===----------------------------------------------------------------------===//
// DominanceInfoBase
//===----------------------------------------------------------------------===//

bool isRunRegionOp(Operation *op) { return false; }

void processRegionDom(
    DT *dt, DenseMap<Region *, std::pair<DTNode *, DTNode *>> &R2EntryExit,
    DenseMap<mlir::Block *, std::pair<DTNode *, DTNode *>> &Block2Node,
    DenseMap<Operation *, DTNode *> &Op2Node, Region *r);

void processOpDom(
    DT *dt, DenseMap<Region *, std::pair<DTNode *, DTNode *>> &R2EntryExit,
    DenseMap<mlir::Block *, std::pair<DTNode *, DTNode *>> &Block2Node,
    DenseMap<Operation *, DTNode *> &Op2Node, Operation *op) {
  const int numRegions = op->getNumRegions();
  for (int i = 0; i < numRegions; i++) {
    Region &R = op->getRegion(i);
    processRegionDom(dt, R2EntryExit, Block2Node, Op2Node, &R);
  }
}

void processRegionPostDom(
    DT *dt, DenseMap<Region *, std::pair<DTNode *, DTNode *>> &R2EntryExit,
    DenseMap<mlir::Block *, std::pair<DTNode *, DTNode *>> &Block2Node,
    DenseMap<Operation *, DTNode *> &Op2Node, Region *r);

void processOpPostDom(
    DT *dt, DenseMap<Region *, std::pair<DTNode *, DTNode *>> &R2EntryExit,
    DenseMap<mlir::Block *, std::pair<DTNode *, DTNode *>> &Block2Node,
    DenseMap<Operation *, DTNode *> &Op2Node, Operation *op) {
  const int numRegions = op->getNumRegions();
  for (int i = 0; i < numRegions; i++) {
    Region &R = op->getRegion(i);
    processRegionPostDom(dt, R2EntryExit, Block2Node, Op2Node, &R);
  }
}



void getRegionsFromValue(Value v, std::set<mlir::Region *> &out) {
  assert(!v.dyn_cast<BlockArgument>() &&
         "region value cannot be block argument!");

  if (RgnValOp val = v.getDefiningOp<RgnValOp>()) {
    out.insert(&val.getRegion());
  } else {
    Operation *op = v.getDefiningOp();
    assert(op && "value that is not a block argument must be an op!");
    for (mlir::Value operand : op->getOperands()) {
      getRegionsFromValue(operand, out);
    }
  }
}


void processRegionPostDom(
    DT *dt, DenseMap<Region *, std::pair<DTNode *, DTNode *>> &R2EntryExit,
    DenseMap<mlir::Block *, std::pair<DTNode *, DTNode *>> &Block2Node,
    DenseMap<Operation *, DTNode *> &Op2Node, mlir::Region *R) {

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
  DTNode *RegionExit = DTNode::newExit(R, dt);
  dt->Nodes.push_back(RegionExit);

  R2EntryExit[R] = {RegionEntry, RegionExit};

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
        RegionExit->addSuccessor(ThisExit);
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
}



void processRegionDom(
    DT *dt, DenseMap<Region *, std::pair<DTNode *, DTNode *>> &R2EntryExit,
    DenseMap<mlir::Block *, std::pair<DTNode *, DTNode *>> &Block2Node,
    DenseMap<Operation *, DTNode *> &Op2Node, mlir::Region *R) {

  if (R->getBlocks().size() == 0) { return; }

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
  DTNode *RegionExit = DTNode::newExit(R, dt);
  dt->Nodes.push_back(RegionExit);

  R2EntryExit[R] = {RegionEntry, RegionExit};

  for (mlir::Block &B : *R) {
    for (Operation &Op : B) {
      // recursively process regions.
      processOpDom(dt, R2EntryExit, Block2Node, Op2Node, &Op);

      Op2Node[&Op] = Block2Node[&B].second;
      /*
      if (RgnCallValOp call = mlir::dyn_cast<RgnCallValOp>(Op)) {
        // need over-approximation of all regions!
        // call have call(select(cond, region_l, region_r))
        std::set<Region *> calledRegions;
        getRegionsFromValue(call.getFn(), calledRegions);

        DTNode *Cur = Block2Node[&B].second;
        DTNode *Next = DTNode::newOp(call, dt);
        dt->Nodes.push_back(Next);
        for (Region *R : calledRegions) {
          DTNode *CallEntry = R2EntryExit[R].first;
          DTNode *CallExit = R2EntryExit[R].second;

          Cur->addSuccessor(CallEntry);
          CallExit->addSuccessor(Next);
        }
        // this is now new final node in block.
        Block2Node[&B].second = Next;
        continue;
      }

      if (RgnJumpValOp jmp = mlir::dyn_cast<RgnJumpValOp>(Op)) {
        RgnValOp val = jmp.getFn().getDefiningOp<RgnValOp>();
        assert(val && "expected RgnJmpVal to point to valid RgnVal");
        // TODO: handle loops!

        // this is now new final node in block.
        DTNode *CallEntry = R2EntryExit[&val.getRegion()].first;
        DTNode *CallExit = R2EntryExit[&val.getRegion()].second;

        DTNode *Cur = Block2Node[&B].second;
        // DTNode *Next = DTNode::newOp(jmp);

        Cur->addSuccessor(CallEntry);
        // CallExit->addSuccessor(Next);
        // vv THINK: does this matter?
        Block2Node[&B].second = CallExit;
        continue;

      }
      */

      // return like op. exit to region exit.
      if (Op.hasTrait<OpTrait::IsTerminator>() &&
          Op.hasTrait<OpTrait::ReturnLike>()) {
        // add edit to exit block of region
        DTNode *ThisExit = Block2Node[&B].second;
        ThisExit->addSuccessor(RegionExit);
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
          ThisExit->addSuccessor(NextEntry);
        }
        continue;
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

  op->walk([&](mlir::FuncOp f) {
    DT *dt = new DT();
    auto dominanceInfo = std::make_unique<base>();

    if (!IsPostDom) {
      processOpDom(dt, this->R2EntryExit, this->Block2EntryExit, this->Op2Node,
                   f);
      assert(this->R2EntryExit.count(&f.getRegion()));
      dt->entry = this->R2EntryExit[&f.getRegion()].first;
      assert(dt->entry);
      llvm::errs() << "trying recalculate...\n";
      // dominanceInfo->recalculate(*dt);
      llvm::errs() << "\n\tSUCCESS!\n";
      if (DEBUG) {
        llvm::dbgs() << "\n\n@@@@processing function.. |" << f << "\n";
      }
    } else {
      processOpPostDom(dt, this->R2EntryExit, this->Block2EntryExit, this->Op2Node,
                   f);
      dt->entry = this->R2EntryExit[&f.getRegion()].second;
      dominanceInfo->recalculate(*dt);
      if (DEBUG) {
        llvm::dbgs() << "\n\n@@@@processing function.. |" << f << "\n";
      }
    }

    
    for (int i = 0; i < dt->Nodes.size(); ++i) {
      dt->Nodes[i]->Info = dominanceInfo.get();
      dt->Nodes[i]->DebugIndex = i;
    }

    if (DEBUG) {

      for (int i = 0; i < dt->Nodes.size(); ++i) {
        for (int j = 0; j < dt->Nodes.size(); ++j) {
          llvm::dbgs() << "dominates(" << i << " " << j
                       << ", isPostDom:" << IsPostDom << ") = "
                       << dominanceInfo->dominates(dt->Nodes[i], dt->Nodes[j])
                       << "\n";
        }
      }

      for (int i = 0; i < dt->Nodes.size(); ++i) {
        llvm::dbgs() << "\n##" << i
                     << (dt->entry == dt->Nodes[i] ? " ENTRY" : "") << ":\n";
        dt->Nodes[i]->print(llvm::dbgs());
        llvm::dbgs() << "\n==\n";
      }

      std::map<DTNode *, int> node2ix;
      for (int i = 0; i < dt->Nodes.size(); ++i) {
        node2ix[dt->Nodes[i]] = i;
      }

      llvm::dbgs() << "edges:\n";
      for (int i = 0; i < dt->Nodes.size(); ++i) {
        for (int j = 0; j < dt->Nodes[i]->successors.size(); ++j) {
          llvm::dbgs() << i << " -> " << node2ix[dt->Nodes[i]->successors[j]]
                       << "\n";
        }
      }

      int FD;
      llvm::sys::fs::openFileForWrite(
          "/home/bollu/temp/" + f.getName() + "-graph.dot", FD);
      llvm::raw_fd_ostream O(FD, /*shouldClose=*/true);
      llvm::WriteGraph(O, dt);
    }
    func2Dominance.insert({f.getOperation(), std::move(dominanceInfo)});
  });

  // this->dt = new DT();

  // processOpDom(this->dt, this->R2EntryExit, this->Block2EntryExit,
  // this->Op2Node,
  //           op);
  // this->dt->entry = this->R2EntryExit[&module.getRegion()].first;

  // op->walk([&](Operation *op) {
  //   const int numRegions = op->getNumRegions();
  //   for (int i = 0; i < numRegions; i++) {
  //     Region &R = op->getRegion(i);
  //     processRegionDom(R2EntryExit, Block2Node, &R);
  //   }
  // });

  // std::unique_ptr<llvm::DominatorTreeBase<DTNode, IsPostDom>> opDominance =
  // this->dominanceInfo = std::make_unique<base>();

  // int FD;
  // llvm::sys::fs::openFileForWrite("/home/bollu/temp/graph.dot", FD);
  // llvm::raw_fd_ostream O(FD, /*shouldClose=*/ true);
  // llvm::WriteGraph(O, this->dt);

  // dominanceInfo->recalculate(*this->dt);

  // disconnected nodes dominate each other?!

  // DTNode *Entry = R2EntryExit[op].first;
  // DTNode *Entry;
  // opDominance->recalculate(*this);
  // dominanceInfos.try_emplace(&region, std::move(opDominance));

  // dominanceInfos.clear();

  // // Build the dominance for each of the operation regions.
  // op->walk([&](Operation *op) {
  //   auto kindInterface = dyn_cast<RegionKindInterface>(op);
  //   unsigned numRegions = op->getNumRegions();
  //   for (unsigned i = 0; i < numRegions; i++) {
  //     Region &region = op->getRegion(i);
  //     // Don't compute dominance if the region is empty.
  //     if (region.empty())
  //       continue;

  //     // Dominance changes pased on the region type. Avoid the helper
  //     // function here so we don't do the region cast repeatedly.
  //     bool hasSSADominance =
  //         op->isRegistered() &&
  //         (!kindInterface || kindInterface.hasSSADominance(i));
  //     // If a region has SSADominance, then compute detailed dominance
  //     // info.  Otherwise, all values in the region are live anywhere
  //     // in the region, which is represented as an empty entry in the
  //     // dominanceInfos map.
  //     if (hasSSADominance) {
  //       auto opDominance = std::make_unique<base>();
  //       opDominance->recalculate(region);
  //       dominanceInfos.try_emplace(&region, std::move(opDominance));
  //     }
  //   }
  // });
}

/// Walks up the list of containers of the given block and calls the
/// user-defined traversal function for every pair of a region and block that
/// could be found during traversal. If the user-defined function returns true
/// for a given pair, traverseAncestors will return the current block. Nullptr
/// otherwise.
template <typename FuncT>
Block *traverseAncestors(Block *block, const FuncT &func) {
  assert(false && "unimplemented");
  // Invoke the user-defined traversal function in the beginning for the current
  // block.
  if (func(block))
    return block;

  Region *region = block->getParent();
  while (region) {
    Operation *ancestor = region->getParentOp();
    // If we have reached to top... return.
    if (!ancestor || !(block = ancestor->getBlock()))
      break;

    // Update the nested region using the new ancestor block.
    region = block->getParent();

    // Invoke the user-defined traversal function and check whether we can
    // already return.
    if (func(block))
      return block;
  }
  return nullptr;
}

/// Tries to update the given block references to live in the same region by
/// exploring the relationship of both blocks with respect to their regions.
static bool tryGetBlocksInSameRegion(Block *&a, Block *&b) {
  assert(false && "unimplemented");

  // If both block do not live in the same region, we will have to check their
  // parent operations.
  if (a->getParent() == b->getParent())
    return true;

  // Iterate over all ancestors of a and insert them into the map. This allows
  // for efficient lookups to find a commonly shared region.
  llvm::SmallDenseMap<Region *, Block *, 4> ancestors;
  traverseAncestors(a, [&](Block *block) {
    ancestors[block->getParent()] = block;
    return false;
  });

  // Try to find a common ancestor starting with regionB.
  b = traverseAncestors(
      b, [&](Block *block) { return ancestors.count(block->getParent()) > 0; });

  // If there is no match, we will not be able to find a common dominator since
  // both regions do not share a common parent region.
  if (!b)
    return false;

  // We have found a common parent region. Update block a to refer to this
  // region.
  auto it = ancestors.find(b->getParent());
  assert(it != ancestors.end());
  a = it->second;
  return true;
}

template <bool IsPostDom>
Block *
DominanceInfoBase<IsPostDom>::findNearestCommonDominator(Block *a,
                                                         Block *b) const {
  if (DEBUG) {
    llvm::dbgs() << __PRETTY_FUNCTION__ << "\n";
    llvm::dbgs() << "\n-a(?)\n";
    a->print(llvm::dbgs());
    llvm::dbgs() << "\n-b(?)\n";
    b->print(llvm::dbgs());
  }
  if (!a || !b) {
    return nullptr;
  }
  if (a == b) {
    return a;
  }

  // assert(false && "unimplemented");

  auto ita = this->Block2EntryExit.find(a);
  auto itb = this->Block2EntryExit.find(b);

  // we are sometimes asked about the block which is the entry block of the
  // **module region**. This is a nonsensical BB to be queried about, so ignore
  // it.
  if (ita == Block2EntryExit.end() || itb == Block2EntryExit.end()) {
    return nullptr;
  }

  assert(ita != Block2EntryExit.end());
  assert(itb != Block2EntryExit.end());

  if (DEBUG) {

    llvm::dbgs() << "\n-a(" << ita->second.first->DebugIndex << ")\n";
    a->print(llvm::dbgs());
    llvm::dbgs() << "\n-b(" << itb->second.first->DebugIndex << ")\n";
    b->print(llvm::dbgs());
  }
  if (a != b) {
    assert(ita->second.first != itb->second.first);
  }

  assert(ita->second.first->Info == itb->second.first->Info &&
         "both must have same dom info data structure");
  base *dominanceInfo = (base *)ita->second.first->Info;

  // TODO, HACK, SID: this assumes we return the entry block!
  // This is totally bankrupt in our case.
  //  Find a correct thing to return.

  // check if entry of A properly dominates entry of B.
  // Operation *fn = a->getParentOp()->getParentOfType<FuncOp>();
  // if (fn == nullptr) { return false;}

  // llvm::dbgs() << "parent: " << fn << "\n";
  // auto domit = func2Dominance.find(fn);
  // assert(domit != func2Dominance.end());
  // check if my exit dominates your entry.
  DTNode *commonDom = dominanceInfo->findNearestCommonDominator(
      ita->second.first, itb->second.first);

  if (DEBUG) {
    llvm::dbgs() << "-commonDom: ";
    if (commonDom) {
      commonDom->print(llvm::dbgs());
    } else {
      llvm::dbgs() << "nullptr ";
    }
    llvm::dbgs() << "\n";
  }
  if (!commonDom) {
    return nullptr;
  }

  assert(commonDom);
  assert(commonDom->kind == DTNode::Kind::DTBlock);

  return commonDom->getBlock();

  // // If either a or b are null, then conservatively return nullptr.
  // if (!a || !b)
  //   return nullptr;

  // // Try to find blocks that are in the same region.
  // if (!tryGetBlocksInSameRegion(a, b))
  //   return nullptr;

  // // Get and verify dominance information of the common parent region.
  // Region *parentRegion = a->getParent();
  // auto infoAIt = dominanceInfos.find(parentRegion);
  // if (infoAIt == dominanceInfos.end())
  //   return nullptr;

  // // Since the blocks live in the same region, we can rely on already
  // // existing dominance functionality.
  // return infoAIt->second->findNearestCommonDominator(a, b);
}

template <bool IsPostDom>
DominanceInfoNode *DominanceInfoBase<IsPostDom>::getNode(Block *a) {
  assert(false && "unimplemented");
  // Region *region = a->getParent();
  // assert(dominanceInfos.count(region) != 0);
  // return dominanceInfos[region]->getNode(a);
}

/// Return true if the specified block A properly dominates block B.
template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::properlyDominatesBB(Block *a,
                                                       Block *b) const {

  if (DEBUG) {
    llvm::dbgs() << __PRETTY_FUNCTION__ << "\n";
  }
  // If either a or b are null, then conservatively return false.
  if (!a || !b) {
    return false;
  }
  if (DEBUG) {

    llvm::dbgs() << "\n-a(?)\n";
    a->print(llvm::dbgs());
    llvm::dbgs() << "\n-b(?)\n";
    b->print(llvm::dbgs());
  }
  auto ita = this->Block2EntryExit.find(a);
  auto itb = this->Block2EntryExit.find(b);

  // we are sometimes asked about blocks that are like the module's
  // entry block which is nonsensical. We hold no information about such blocks,
  // so give up and conservatively return false.
  if (ita == this->Block2EntryExit.end() ||
      itb == this->Block2EntryExit.end()) {
    return false;
  }
  assert(ita != Block2EntryExit.end());
  assert(itb != Block2EntryExit.end());

  if (DEBUG) {
    llvm::dbgs() << "\n-a(" << ita->second.first->DebugIndex << ")\n";
    a->print(llvm::dbgs());
    llvm::dbgs() << "\n-b(" << itb->second.first->DebugIndex << ")\n";
    b->print(llvm::dbgs());
  }

  if (a != b) {
    assert(ita->second.first != itb->second.first);
  }

  assert(ita->second.first->Info == itb->second.first->Info &&
         "both must have same dom info data structure");
  base *dominanceInfo = (base *)ita->second.first->Info;

  // check if entry of A properly dominates entry of B.
  // Operation *fn = a->getParentOp()->getParentOfType<FuncOp>();
  // if (fn == nullptr) { return false;}

  // llvm::dbgs() << "parent: " << fn << "\n";
  // auto domit = func2Dominance.find(fn);
  // assert(domit != func2Dominance.end());
  // check if my exit dominates your entry.
  return dominanceInfo->properlyDominates(ita->second.second,
                                          itb->second.first);

  // return dominanceInfo->properlyDominates(ita->second.first,
  // itb->second.first);

  assert(false && "unimplemented");

  if (a->getParent() == b->getParent()) {
    // A block dominates itself but does not properly dominate itself.
    if (a == b) {
      return b;
    }
  } else {
    // These blocks are in different regions.
    // if a's region does not contain b's region, then a does not dominates b.
    if (!a->getParent()->isProperAncestor(b->getParent())) {
      return false;
    }

    // a's region is the parent of b's region.
    // check if a ever calls `run` on `b`.
  }

  // // A block dominates itself but does not properly dominate itself.
  // if (a == b)
  //   return false;

  // // If either a or b are null, then conservatively return false.
  // if (!a || !b)
  //   return false;

  // // if both blocks are not in the same region, then 'a' properly dominates
  // // 'b' if 'a' executes 'b'.

  // // If both blocks are not in the same region, 'a' properly dominates 'b' if
  // // 'b' is defined in an operation region that (recursively) ends up being
  // // dominated by 'a'. Walk up the list of containers enclosing B.
  // const Region *regionA = a->getParent();
  // if (regionA != b->getParent()) {
  //   b = traverseAncestors(
  //       b, [&](Block *block) { return block->getParent() == regionA; });

  //   // If we could not find a valid block b then it is a not a dominator.
  //   if (!b)
  //     return false;

  //   // Check to see if the ancestor of 'b' is the same block as 'a'.
  //   if (a == b)
  //     return true;
  // }

  // // Otherwise, use the standard dominance functionality.

  // // If we don't have a dominance information for this region, assume that b
  // is
  // // dominated by anything.
  // auto baseInfoIt = dominanceInfos.find(regionA);
  // if (baseInfoIt == dominanceInfos.end())
  //   return true;
  // return baseInfoIt->second->properlyDominates(a, b);
}

/// Return true if the specified block is reachable from the entry block of its
/// region.
template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::isReachableFromEntry(Block *a) const {
  if (DEBUG) {
    llvm::dbgs() << __PRETTY_FUNCTION__ << "\n";
    llvm::dbgs() << "-a: " << a << "\n";
    llvm::dbgs() << "-a: ";
    a->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }

  // assert(false && "unreachable");
  auto it = this->Block2EntryExit.find(a);
  // TODO: why can there a BB we don't understand? I don't get it, but OK.
  // Sid: this condition was stolen from previous code.
  if (it == this->Block2EntryExit.end()) {
    return true;
  }

  assert(it != this->Block2EntryExit.end());
  DTNode *Entry = it->second.first;
  base *dominanceInfo = (base *)Entry->Info;
  return dominanceInfo->isReachableFromEntry(Entry);
  // Region *regionA = a->getParent();
  // auto baseInfoIt = dominanceInfos.find(regionA);
  // if (baseInfoIt == dominanceInfos.end()) {
  //   return true;
  // }

  // // return
  // baseInfoIt->second->isReachableFromEntry(this->Block2EntryExit[a].first);
}

template class detail::DominanceInfoBase</*IsPostDom=*/true>;
template class detail::DominanceInfoBase</*IsPostDom=*/false>;

//===----------------------------------------------------------------------===//
// DominanceInfo
//===----------------------------------------------------------------------===//

/// Return true if operation A properly dominates operation B.
bool DominanceInfo::properlyDominatesOO(Operation *a, Operation *b) const {
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
    return dominatesOO(a, bAncestor);
  }

  // If the blocks are different, check if a's block dominates b's.
  return properlyDominatesBB(aBlock, bBlock);
}

/// Return true if value A properly dominates operation B.
bool DominanceInfo::properlyDominates(Value a, Operation *b) const {
  if (Operation *aOp = a.getDefiningOp()) {
    auto ita = this->Op2Node.find(aOp);
    auto itb = this->Op2Node.find(b);
    assert(ita != Op2Node.end());
    assert(itb != Op2Node.end());

    assert(ita->second->Info == itb->second->Info &&
           "both must have same dom info data structure");
    base *dominanceInfo = (base *)ita->second->Info;

    if (ita->second == itb->second) {
      // these are next to each other in the same BB / are not interleaved with
      // a region instruction.
      return aOp != b && aOp->isBeforeInBlock(b);
    }

    return aOp != b && dominanceInfo->dominates(ita->second, itb->second);
  }

  assert(a.isa<BlockArgument>() && "value must be op or block argument");
  // assert(false && "invoking block argument check!");
  // block arguments properly dominate all operations in their own block, so
  // we use a dominates check here, not a properlyDominates check.
  return dominatesBB(a.cast<BlockArgument>().getOwner(), b->getBlock());

  assert(false && "unimplemented");

  if (auto *aOp = a.getDefiningOp()) {
    // Dominance changes based on the region type.
    auto *aRegion = aOp->getParentRegion();
    unsigned aRegionNum = aRegion->getRegionNumber();
    Operation *ancestor = aRegion->getParentOp();
    // Dominance changes based on the region type. In a region with SSA
    // dominance, values defined by an operation cannot be used by the
    // operation. In other regions kinds they can be used the operation.
    if (hasSSADominance(ancestor, aRegionNum)) {
      // The values defined by an operation do *not* dominate any nested
      // operations.
      if (aOp->getParentRegion() != b->getParentRegion() && aOp->isAncestor(b))
        return false;
    }
    return properlyDominatesOO(aOp, b);
  }

  // block arguments properly dominate all operations in their own block, so
  // we use a dominates check here, not a properlyDominates check.
  return dominatesBB(a.cast<BlockArgument>().getOwner(), b->getBlock());
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
  return properlyDominatesBB(aBlock, bBlock);
}
