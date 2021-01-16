// RUN: mlir-opt %s 

func @main() {
    %c1ix = constant 1 : index
    %c0f32 = constant 0.0 : f32
    scf.parallel (%iv) = (%c1ix) to (%c1ix) step (%c1ix) init (%c0f32) -> f32 {
            // %use_before_def does not in fact exist. That this crashes is incorrect.
            // It should instead error out saying that `%use_before_def` does not 
            // have a def before its use.
            scf.reduce(%use_before_def) : f32 {
                ^bb0(%use_before_def : f32, %_: f32):
                     scf.reduce.return %_ : f32
            }
    }
    return
}
