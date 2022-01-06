{ pkgs ? import <nixpkgs> {} }:

with pkgs;

stdenv.mkDerivation {
	name = "mlir-llvm-project";
	src = ./.;
	nativeBuildInputs = [cmake ninja clang lld python3];
	cmakeFlags = ''
		   -G Ninja  ../llvm/
		   -DLLVM_ENABLE_PROJECTS=mlir 
		   -DLLVM_BUILD_EXAMPLES=ON 
		   -DCMAKE_BUILD_TYPE=RelWithDebInfo 
		   -DLLVM_ENABLE_ASSERTIONS=ON 
		   -DBUILD_SHARED_LIBS=ON
		   -DLLVM_INSTALL_UTILS=ON 
		   -DLLVM_ENABLE_ASSERTIONS=ON 
	   	   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON 
                   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
		'';
	buildPhase = ''ninja'';
}

