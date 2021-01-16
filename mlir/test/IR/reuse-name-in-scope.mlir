// RUN: mlir-opt %s 

func @main() {
    %c0ix = constant 0 : index
    %c0f32 = constant 0.0 : f32
    scf.parallel (%iv) = (%c0ix) to (%c0ix) step (%c0ix) init (%c0f32) -> f32 {
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
