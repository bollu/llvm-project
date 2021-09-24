//===- CSE.cpp - Common Sub-expression Elimination ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs a simple common sub-expression elimination
// algorithm on operations within a region.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/RecyclingAllocator.h"
#include <deque>
#include <queue>
using namespace mlir;

namespace {
struct SimpleOperationInfo : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    return OperationEquivalence::computeHash(
        const_cast<Operation *>(opC),
        /*hashOperands=*/OperationEquivalence::directHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
  }
  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return OperationEquivalence::isEquivalentTo(
        const_cast<Operation *>(lhsC), const_cast<Operation *>(rhsC),
        /*mapOperands=*/OperationEquivalence::exactValueMatch,
        /*mapResults=*/OperationEquivalence::ignoreValueEquivalence,
        OperationEquivalence::IgnoreLocations);
  }
};
} // end anonymous namespace

namespace {
/// Simple common sub-expression elimination.
struct CSE : public CSEBase<CSE> {
  /// Shared implementation of operation elimination and scoped map definitions.
  using AllocatorTy = llvm::RecyclingAllocator<
      llvm::BumpPtrAllocator,
      llvm::ScopedHashTableVal<Operation *, Operation *>>;
  using ScopedMapTy = llvm::ScopedHashTable<Operation *, Operation *,
                                            SimpleOperationInfo, AllocatorTy>;

  /// Represents a single entry in the depth first traversal of a CFG.
  struct CFGStackNode {
    CFGStackNode(ScopedMapTy &knownValues, DominanceInfoNode *node)
        : scope(knownValues), node(node), childIterator(node->begin()),
          processed(false) {}

    /// Scope for the known values.
    ScopedMapTy::ScopeTy scope;

    DominanceInfoNode *node;
    DominanceInfoNode::const_iterator childIterator;

    /// If this node has been fully processed yet or not.
    bool processed;
  };

  /// Attempt to eliminate a redundant operation. Returns success if the
  /// operation was marked for removal, failure otherwise.
  void simplifyOperation(ScopedMapTy &knownValues, Operation &op);
  void simplifyBlock(ScopedMapTy &knownValues, Block *bb, bool hasSSADominance);
  void simplifyRegion(ScopedMapTy &knownValues, Region &region);
  void simplifyDomNode(ScopedMapTy &knownValues, llvm::DomTreeNodeBase<DTNode> *node);

  void runOnOperation() override;

private:
  /// Operations marked as dead and to be erased.
  std::vector<Operation *> opsToErase;
  DominanceInfo *domInfo = nullptr;
};
} // end anonymous namespace

/// Attempt to eliminate a redundant operation.
void CSE::simplifyOperation(ScopedMapTy &knownValues, Operation &op) {
  if (op.hasTrait<OpTrait::IsTerminator>()) {
    return;
  }
  // If the operation is already trivially dead just add it to the erase
  // list.
  if (isOpTriviallyDead(&op)) {
    opsToErase.push_back(&op);
    return;
  }

  if (op.getNumRegions() > 0) { return; }

  // Don't simplify operations with nested blocks. We don't currently model
  // equality comparisons correctly among other things. It is also unclear
  // whether we would want to CSE such operations.
  // assert(op.getNumRegions() == 0);

  // TODO: We currently only eliminate non side-effecting
  // operations.
  if (!MemoryEffectOpInterface::hasNoEffect(&op)) {
    return;
  }
  // Look for an existing definition for the operation.
  auto *existing = knownValues.lookup(&op);
  if (!existing) {
    knownValues.insert(&op, &op);
    return;
  }
  // If we find one then replace all uses of the current operation with the
  // existing one and mark it for deletion. We can only replace an operand
  // in an operation if it has not been visited yet. If the region has SSA
  // dominance, then we are guaranteed to have not visited any use of the
  // current operation
  // llvm::errs() << "###replacing\n";
  // llvm::errs() << "\t" << op;
  // llvm::errs() << "\n\tby: " << *existing;
  // llvm::errs() << "\n==\n";
  // getchar();

  assert(&op != existing);
  op.replaceAllUsesWith(existing);
  opsToErase.push_back(&op);
}

void CSE::simplifyBlock(ScopedMapTy &knownValues, Block *bb,
                        bool hasSSADominance) {
  assert(false);
  // for (auto &op : *bb) {
  //   // If the operation is simplified, we don't process any held regions.
  //   if (succeeded(simplifyOperation(knownValues, &op, hasSSADominance)))
  //     continue;

  //   // Most operations don't have regions, so fast path that case.
  //   if (op.getNumRegions() == 0)
  //     continue;

  //   // If this operation is isolated above, we can't process nested regions
  //   with
  //   // the given 'knownValues' map. This would cause the insertion of
  //   implicit
  //   // captures in explicit capture only regions.
  //   if (op.mightHaveTrait<OpTrait::IsIsolatedFromAbove>()) {
  //     ScopedMapTy nestedKnownValues;
  //     for (auto &region : op.getRegions())
  //       simplifyRegion(nestedKnownValues, region);
  //     continue;
  //   }

  //   // Otherwise, process nested regions normally.
  //   for (auto &region : op.getRegions())
  //     simplifyRegion(knownValues, region);
  // }
}

void CSE::simplifyRegion(ScopedMapTy &knownValues, Region &region) {
  assert(false);
  // // If the region is empty there is nothing to do.
  // if (region.empty())
  //   return;

  // const bool hasSSADom = this->domInfo->hasSSADominance(region);
  // // bool hasSSADominance = false; /*conservative*/

  // // If the region only contains one block, then simplify it directly.
  // if (region.hasOneBlock()) {
  //   ScopedMapTy::ScopeTy scope(knownValues);
  //   simplifyBlock(knownValues, &region.front(), hasSSADom);
  //   return;
  // }

  // // If the region does not have dominanceInfo, then skip it.
  // // TODO: Regions without SSA dominance should define a different
  // // traversal order which is appropriate and can be used here.
  // if (!hasSSADom)
  //   return;

  // // Note, deque is being used here because there was significant
  // performance
  // // gains over vector when the container becomes very large due to the
  // // specific access patterns. If/when these performance issues are no
  // // longer a problem we can change this to vector. For more information
  // see
  // // the llvm mailing list discussion on this:
  // //
  // http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20120116/135228.html
  // std::deque<std::unique_ptr<CFGStackNode>> stack;

  // // Process the nodes of the dom tree for this region.
  // stack.emplace_back(std::make_unique<CFGStackNode>(
  //     knownValues, domInfo->getRootNode(&region)));

  // // assert(false && "should have died here -_^");

  // while (!stack.empty()) {
  //   auto &currentNode = stack.back();

  //   // Check to see if we need to process this node.
  //   if (!currentNode->processed) {
  //     currentNode->processed = true;
  //     // Block *currentBlock = currentNode->node->getBlock();
  //     Block *currentBlock = nullptr;
  //     assert(false && "should crash here!");
  //     simplifyBlock(knownValues, currentBlock, hasSSADom);
  //   }

  //   // Otherwise, check to see if we need to process a child node.
  //   if (currentNode->childIterator != currentNode->node->end()) {
  //     auto *childNode = *(currentNode->childIterator++);
  //     stack.emplace_back(
  //         std::make_unique<CFGStackNode>(knownValues, childNode));
  //   } else {
  //     // Finally, if the node and all of its children have been processed
  //     // then we delete the node.
  //     stack.pop_back();
  //   }
  // }
}

void CSE::simplifyDomNode(ScopedMapTy &knownValues, llvm::DomTreeNodeBase<DTNode> *node) {
  llvm::errs() << "at Domtree node: " << node << "\n";
  ScopedMapTy::ScopeTy scope(knownValues);
  // recall that this is the "slow" implmentation where EACH op has an exit.
  DTNode *dtnode = node->getBlock();
  if (dtnode->kind == DTNode::Kind::DTOpExit) {
      auto it = dtnode->getOp();
    llvm::errs() << "visiting op exit\n";
    dtnode->getOp()->print(llvm::errs());
    llvm::errs() << "\n---\n";
    // getchar();

    this->simplifyOperation(knownValues, *it);
  }
  for (llvm::DomTreeNodeBase<DTNode> *succ : node->children() ) {
    simplifyDomNode(knownValues, succ);
  }

  // // TODO: convert to BFS, since we can now have "islands" of DAGs.
  // // We now have a dominator DAG.
  // static int scopeCount = 0;
  // // always open a new scope?
  // ScopedMapTy::ScopeTy scope(knownValues);
  // ++scopeCount;
  // DTNode *dtnode = node->getBlock();
  // if (dtnode->kind == DTNode::Kind::DTBlock) {

  //   llvm::errs() << "visiting BB[" << scopeCount << "]\n";
  //   dtnode->getBlock()->print(llvm::errs());
  //   llvm::errs() << "\n---\n";
  //   getchar();

  //   for (auto &op : *dtnode->getBlock()) {
  //     bool hasSSADom = true;
  //     if (!hasSSADom) {
  //       continue;
  //     }
  //     // operations with regions are handled after their exit.
  //     if (op.getNumRegions() != 0) {
  //       break;
  //     }

  //     this->simplifyOperation(knownValues, op);
  //   } // end loop over ops
  // } else if (dtnode->kind == DTNode::Kind::DTOpExit) {
  //   // can now process eveyrthing after the op.
  //   // TODO: consider hash-consing the op itself!

  //   // start *after* this op.

  //   llvm::errs() << "visiting op exit[" << scopeCount << "]\n";
  //   dtnode->getOp()->print(llvm::errs());
  //   llvm::errs() << "\n---\n";
  //   getchar();

  //   auto it = dtnode->getOp()->getIterator();
  //   ++it;
  //   for (it; it != dtnode->getOp()->getBlock()->end(); ++it) {
  //     // process ops till the next op with a region.
  //     bool hasSSADom = true;
  //     if (!hasSSADom) {
  //       continue;
  //     }
  //     // operations with regions are handled after their exit.
  //     if (it->getNumRegions() != 0) {
  //       break;
  //     }

  //     this->simplifyOperation(knownValues, *it);
  //   }
  // }

  // // TODO: use BFS, not DFS. That way, if we have an op with many regions, we
  // // will complete all the regions before navigating to the OpExit! op1 ->
  // // [r1, r2] -> op1exit -> op2. we want to have completed processing r1, r2,
  // // beore we being processing op2.
  // scopeCount--;
};

void CSE::runOnOperation() {

  /// A scoped hash table of defining operations within a region.
  ScopedMapTy knownValues;

  domInfo = &getAnalysis<DominanceInfo>();
  Operation *rootOp = getOperation();

  // std::map<llvm::DomTreeNodeBase<DTNode> *, int> visitCount;
  std::set<llvm::DomTreeNodeBase<DTNode> *> visited;
  std::queue<llvm::DomTreeNodeBase<DTNode> *> bfs;
  auto nodes = domInfo->getRootNodes();
  // bfs.push(domInfo->getRootNode());

  for(llvm::DomTreeNodeBase<DTNode> * node : nodes) {
    assert(node);
    llvm::errs() << "root node: |" << node << "|\n";
     this->simplifyDomNode(knownValues, node);

  }


  // while(!bfs.empty()) {
  //   llvm::DomTreeNodeBase<DTNode> *cur = bfs.front();
  //   bfs.pop();
  // //   assert(visited.count(cur) == 0);
  //   if (visited.count(cur)) {
  //     llvm::errs() << "revisiting: " << cur << "\n";
  //     continue;
  //   }
  //   visited.insert(cur);
    // this->simplifyDomNode(knownValues, cur);
  //   for (llvm::DomTreeNodeBase<DTNode> *succ : cur->children() ) {
  //     bfs.push(succ);
  //   }
  // }
  // this->simplifyDomNode(knownValues, domInfo->getRootNode());
  
  // llvm::errs() << "function running CSE on:\n===\n";
  // this->getOperation()->print(llvm::errs());
  // llvm::errs() << "\n===\n";

  /// Erase any operations that were marked as dead during simplification.
  for (auto *op : opsToErase) {
    op->erase();
  }
  opsToErase.clear();

  // We currently don't remove region operations, so mark dominance as
  // preserved.
  markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
  domInfo = nullptr;

  return;

  assert(false && "unreachable");
  for (auto &region : rootOp->getRegions())
    simplifyRegion(knownValues, region);

  // If no operations were erased, then we mark all analyses as preserved.
  if (opsToErase.empty())
    return markAllAnalysesPreserved();

  /// Erase any operations that were marked as dead during simplification.
  for (auto *op : opsToErase)
    op->erase();
  opsToErase.clear();

  // We currently don't remove region operations, so mark dominance as
  // preserved.
  markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
  domInfo = nullptr;
}

std::unique_ptr<Pass> mlir::createCSEPass() { return std::make_unique<CSE>(); }
