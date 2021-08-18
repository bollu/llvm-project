add_mlir_library(MLIRRgn
  RgnDialect.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rgn

  DEPENDS
  MLIRStandardOpsIncGen
  MLIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffine
  MLIRAnalysis
  MLIRCopyOpInterface
  MLIRLinalg
  MLIRLoopLikeInterface
  MLIRMemRef
  MLIRSCF
  MLIRPass
  MLIRTransformUtils
  MLIRVector

  )
