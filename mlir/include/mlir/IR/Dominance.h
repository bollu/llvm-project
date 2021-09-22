//===- Dominance.h - Dominator analysis for CFGs ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DOMINANCE_H
#define MLIR_IR_DOMINANCE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>

extern template class llvm::DominatorTreeBase<mlir::Block, false>;
extern template class llvm::DominatorTreeBase<mlir::Block, true>;

struct DTNode; // forward declare

struct DT {
  DT() {}

  DTNode *entry = nullptr;
  using NodesT = llvm::SmallVector<DTNode *, 4>;
  NodesT Nodes;

  void debug_print(llvm::raw_ostream &o) const;

  DTNode &front() {
    assert(entry);
    return *entry;
  }
};

// look at RegionGraphTraits.h

struct DTNode {
  // void *Info = nullptr; // pointer to dominance info data structure.
  int DebugIndex = 0; // index used for debugging.
  static int Count;

  enum class Kind {
    DTBlock,
    // DTExit,
    DTOpExit,        // node for an operation that implies region semantics.
    DTToplevelEntry, // top level entry node.
  };

  DTNode::Kind kind;

  using SuccessorRange = std::vector<DTNode *>;
  SuccessorRange successors;

  using succ_iterator = SuccessorRange::iterator;
  succ_iterator succ_begin() { return getSuccessors().begin(); }
  succ_iterator succ_end() { return getSuccessors().end(); }
  SuccessorRange &getSuccessors() { return this->successors; }

  using PredecessorRange = std::vector<DTNode *>;
  PredecessorRange predecessors;

  using pred_iterator = PredecessorRange::iterator;
  pred_iterator pred_begin() { return getPredecessors().begin(); }
  pred_iterator pred_end() { return getPredecessors().end(); }
  PredecessorRange &getPredecessors() { return this->predecessors; }

  void addSuccessor(DTNode *&next) {
    this->successors.push_back(next);
    next->predecessors.push_back(this);
  }

  static DTNode *newBlock(mlir::Block *b, DT *parent) {
    DTNode *node = new DTNode(parent);
    node->b = b;
    node->kind = Kind::DTBlock;
    return node;
  }

  // static DTNode *newExit(mlir::Region *r, DT *parent) {
  //   DTNode *node = new DTNode(parent);
  //   node->r = r;
  //   node->kind = Kind::DTExit;
  //   return node;
  // }

  static DTNode *newOpExit(mlir::Operation *op, DT *parent) {
    DTNode *node = new DTNode(parent);
    node->op = op;
    node->kind = Kind::DTOpExit;
    return node;
  }

  static DTNode *newToplevelEntry(mlir::Operation *op, DT *parent) {
    DTNode *node = new DTNode(parent);
    node->op = op;
    node->kind = Kind::DTToplevelEntry;
    return node;
  }

  DT *getParent() { return parent; }

  void printAsOperand(llvm::raw_ostream &os, bool printType = true) {
    os << this->DebugIndex;
    // assert(false && "unimplemented print for DTNode");
  }

  DTNode(const DTNode &other) = default;
  // explicit DTNode() = default;

  void print(llvm::raw_ostream &os) const;

  mlir::Block *getBlock() {
    assert(this->kind == Kind::DTBlock);
    return this->b;
  }

  mlir::Operation *getOp() {
    assert(this->kind == Kind::DTOpExit);
    return this->op;
  }

private:
  DTNode(DT *parent) : parent(parent), DebugIndex(++Count) {
    parent->Nodes.push_back(this);
  }
  DT *parent = nullptr;
  mlir::Block *b = nullptr;
  mlir::Operation *op = nullptr;
};

inline llvm::raw_ostream &operator << (llvm::raw_ostream &os, const DTNode &node) {
  node.print(os); return os;
}

namespace llvm {
template <>
struct GraphTraits<DTNode *> {
  // using Node = ;
  using NodeRef = DTNode *;
  static NodeRef getEntryNode(NodeRef bb) { return bb; }

  using ChildIteratorType = DTNode::succ_iterator;
  static ChildIteratorType child_begin(NodeRef node) {
    return node->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) { return node->succ_end(); }

  static std::string getNodeIdentifierLabel(const DTNode *Node,
                                            const DT *Graph) {
    return std::to_string(Node->DebugIndex);
  }
};

} // namespace llvm

namespace llvm {
template <>
struct GraphTraits<DT *> : public GraphTraits<DTNode *> {
  // refer to call graph
  static DTNode *getEntryNode(DT *dt) { return dt->entry; }

  using nodes_iterator = DT::NodesT::iterator;
  static nodes_iterator nodes_begin(DT *base) { return base->Nodes.begin(); }
  static nodes_iterator nodes_end(DT *base) { return base->Nodes.end(); }


};

template<>
struct DOTGraphTraits<const DT*> : public DefaultDOTGraphTraits {
    explicit DOTGraphTraits(bool isSimple=true) :
      DefaultDOTGraphTraits(isSimple) {}

  static std::string getNodeLabel(const DTNode *Node,
                                  const DT *&Graph) {
    assert(false);
    return std::to_string(Node->DebugIndex);
  }
};

// template<>
// struct DOTGraphTraits<DT*> : public DefaultDOTGraphTraits {
//     explicit DOTGraphTraits(bool isSimple=true) :
//       DefaultDOTGraphTraits(isSimple) {}

//   static std::string getNodeLabel(const DTNode *Node,
//                                   const DT *&Graph) {
//     assert(false);
//     return std::to_string(Node->DebugIndex);
//   }
// };



} // namespace llvm

namespace llvm {
template <>
struct GraphTraits<Inverse<DTNode *>> {
  using ChildIteratorType = DTNode::pred_iterator;
  using NodeRef = DTNode *;
  static NodeRef getEntryNode(Inverse<NodeRef> inverseGraph) {
    return inverseGraph.Graph;
  }
  static inline ChildIteratorType child_begin(NodeRef node) {
    return node->pred_begin();
  }
  static inline ChildIteratorType child_end(NodeRef node) {
    return node->pred_end();
  }

};
}; // namespace llvm

namespace llvm {
template <>
struct GraphTraits<Inverse<DT *>> : public GraphTraits<Inverse<DTNode *>> {
  using GraphType = Inverse<DT *>;
  using NodeRef = DTNode *;

  static NodeRef getEntryNode(Inverse<DT *> dt) { return dt.Graph->entry; }

  using nodes_iterator = DT::NodesT::iterator;
  static nodes_iterator nodes_begin(Inverse<DT *> dt) {
    return nodes_iterator(dt.Graph->Nodes.begin());
  }
  static nodes_iterator nodes_end(Inverse<DT *> dt) {
    return nodes_iterator(dt.Graph->Nodes.end());
  }



};

} // namespace llvm

namespace mlir {
using DominanceInfoNode = llvm::DomTreeNodeBase<DTNode>;
class Operation;

namespace detail {
template <bool IsPostDom>
class DominanceInfoBase {
public:
    using DTBaseT = llvm::DominatorTreeBase<DTNode, IsPostDom>;
protected:
  DTBaseT *tree = nullptr;
  DenseMap<Block *, std::pair<DTNode *, DTNode *>> Block2EntryExit;
  DenseMap<Operation *, DTNode *> Op2Node;
  DenseMap<Region *, std::pair<DTNode *, DTNode *>> R2EntryExit;

  // DTNode *toplevelEntry = nullptr; // TODO: maybe unused
public:
  DominanceInfoBase() {}
  DominanceInfoBase(Operation *op) { recalculate(op); }
  DominanceInfoBase(DominanceInfoBase &&) = default;
  DominanceInfoBase &operator=(DominanceInfoBase &&) = default;

  DominanceInfoBase(const DominanceInfoBase &) = delete;
  DominanceInfoBase &operator=(const DominanceInfoBase &) = delete;

  /// Recalculate the dominance info.
  void recalculate(Operation *op);

  /// Finds the nearest common dominator block for the two given blocks a
  /// and b. If no common dominator can be found, this function will return
  /// nullptr.
  // TODO: this will change, since we can have nearest common dominator to be an
  // Op.
  Block *findNearestCommonDominator(Block *a, Block *b) const;

  /// Return true if there is dominanceInfo for the given region.
  bool hasDominanceInfo(Region *region) {
    return this->tree != nullptr;
    // return dominanceInfos.count(region) != 0;
  }

  bool hasSSADominance(Operation *op, unsigned index);
  bool hasSSADominance(Region &r);

  llvm::DomTreeNodeBase<DTNode> *getRootNode() {
    return this->tree->getRootNode();
  }

  /// Get the root dominance node of the given region.
  DominanceInfoNode *getRootNode(Region *region) {
    auto it = this->R2EntryExit.find(region);
    assert(it != this->R2EntryExit.end());
    DTNode *node = it->second.first;
    return this->tree->getNode(node);
    // assert(false && "unimplemented");
    // return nullptr;
    // assert(dominanceInfos.count(region) != 0);
    // return dominanceInfos[region]->getRootNode();
  }

  /// Return the dominance node from the Region containing block A.
  DominanceInfoNode *getNode(Block *a);



protected:
  using super = DominanceInfoBase<IsPostDom>;

  /// Return true if the specified block A properly dominates block B.
  bool properlyDominates(Block *a, Block *b) const;

  /// Return true if the specified block is reachable from the entry
  /// block of its region.
  bool isReachableFromEntry(Block *a) const;


  bool isReachableFromParentRegion(Block *a) const;

  /// A mapping of regions to their base dominator tree.
  // DenseMap<Region *, std::unique_ptr<DTBaseT>> dominanceInfos;

  // a mapping from parent regions to child regions which they dominate
  // DenseMap<Region *, SmallVector<Region *, 4>> domParent2Children;
  // DenseMap<Region *, Region *> domChild2Parent;

  // DenseMap<Operation *, std::unique_ptr<base>> func2Dominance;

  // std::unique_ptr<base> dominanceInfo;
  // DT *dt;
};
} // end namespace detail

/// A class for computing basic dominance information. Note that this
/// class is aware of different types of regions and returns a
/// region-kind specific concept of dominance. See RegionKindInterface.
class DominanceInfo : public detail::DominanceInfoBase</*IsPostDom=*/false> {
public:
  using super::super;

  /// Return true if the specified block is reachable from the entry block of
  /// its region. In an SSACFG region, a block is reachable from the entry block
  /// if it is the successor of the entry block or another reachable block. In a
  /// Graph region, all blocks are reachable.
  bool isReachableFromEntry(Block *a) const {
    return super::isReachableFromEntry(a);
  }

  bool isReachableFromParentRegion(Block *a) const {
    return super::isReachableFromParentRegion(a);
  }

  /// Return true if operation A properly dominates operation B, i.e. if A and B
  /// are in the same block and A properly dominates B within the block, or if
  /// the block that contains A properly dominates the block that contains B. In
  /// an SSACFG region, Operation A dominates Operation B in the same block if A
  /// preceeds B. In a Graph region, all operations in a block dominate all
  /// other operations in the same block.
  bool properlyDominates(Operation *a, Operation *b) const;

  /// Return true if operation A dominates operation B, i.e. if A and B are the
  /// same operation or A properly dominates B.
  bool dominates(Operation *a, Operation *b) const {
    return a == b || properlyDominates(a, b);
  }

  /// Return true if value A properly dominates operation B, i.e if the
  /// operation that defines A properlyDominates B and the operation that
  /// defines A does not contain B.
  bool properlyDominates(Value a, Operation *b) const;

  /// Return true if operation A dominates operation B, i.e if the operation
  /// that defines A dominates B.
  bool dominates(Value a, Operation *b) const {
    return (Operation *)a.getDefiningOp() == b || properlyDominates(a, b);
  }

  /// Return true if the specified block A dominates block B, i.e. if block A
  /// and block B are the same block or block A properly dominates block B.
  bool dominates(Block *a, Block *b) const {
    return a == b || properlyDominates(a, b);
  }

  // /// Return true if the specified block A properly dominates block B, i.e.: if
  // /// block A contains block B, or if the region which contains block A also
  // /// contains block B or some parent of block B and block A dominates that
  // /// block in that kind of region.  In an SSACFG region, block A dominates
  // /// block B if all control flow paths from the entry block to block B flow
  // /// through block A. In a Graph region, all blocks dominate all other blocks.
  bool properlyDominates(Block *a, Block *b) const {
    return super::properlyDominates(a, b);
  }

  /// Update the internal DFS numbers for the dominance nodes.
  void updateDFSNumbers();
};

/// A class for computing basic postdominance information.
class PostDominanceInfo : public detail::DominanceInfoBase</*IsPostDom=*/true> {
public:
  using super::super;

  /// Return true if the specified block is reachable from the entry
  /// block of its region.
  bool isReachableFromEntry(Block *a) const {
    return super::isReachableFromEntry(a);
  }

  /// Return true if operation A properly postdominates operation B.
  bool properlyPostDominates(Operation *a, Operation *b);

  /// Return true if operation A postdominates operation B.
  bool postDominates(Operation *a, Operation *b) {
    return a == b || properlyPostDominates(a, b);
  }

  /// Return true if the specified block A properly postdominates block B.
  bool properlyPostDominates(Block *a, Block *b) {
    return super::properlyDominates(a, b);
  }

  /// Return true if the specified block A postdominates block B.
  bool postDominates(Block *a, Block *b) {
    return a == b || properlyPostDominates(a, b);
  }
};



} //  end namespace mlir

namespace llvm {

/// DominatorTree GraphTraits specialization so the DominatorTree can be
/// iterated by generic graph iterators.
template <>
struct GraphTraits<mlir::DominanceInfoNode *> {
  using ChildIteratorType = mlir::DominanceInfoNode::const_iterator;
  using NodeRef = mlir::DominanceInfoNode *;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static inline ChildIteratorType child_begin(NodeRef N) { return N->begin(); }
  static inline ChildIteratorType child_end(NodeRef N) { return N->end(); }
};

template <>
struct GraphTraits<const mlir::DominanceInfoNode *> {
  using ChildIteratorType = mlir::DominanceInfoNode::const_iterator;
  using NodeRef = const mlir::DominanceInfoNode *;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static inline ChildIteratorType child_begin(NodeRef N) { return N->begin(); }
  static inline ChildIteratorType child_end(NodeRef N) { return N->end(); }
};

} // end namespace llvm
#endif
