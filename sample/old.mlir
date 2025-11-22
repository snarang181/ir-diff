// old.mlir

module {
  // Naive matmul implementation
  func.func @matmul_naive(%A: memref<4x4xf32>,
                     %B: memref<4x4xf32>,
                     %C: memref<4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32

    // Initialize C to 0
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        memref.store %zero, %C[%i, %j] : memref<4x4xf32>
      }
    }

    // Naive triple loop: C[i,j] += A[i,k] * B[k,j]
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        scf.for %k = %c0 to %c4 step %c1 {
          %a = memref.load %A[%i, %k] : memref<4x4xf32>
          %b = memref.load %B[%k, %j] : memref<4x4xf32>
          %prod = arith.mulf %a, %b : f32
          %cold = memref.load %C[%i, %j] : memref<4x4xf32>
          %cnew = arith.addf %cold, %prod : f32
          memref.store %cnew, %C[%i, %j] : memref<4x4xf32>
        }
      }
    }

    return
  }

  // Simple ReLU kernel
  func.func @relu(%X: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32

    scf.for %i = %c0 to %c16 step %c1 {
      %x = memref.load %X[%i] : memref<16xf32>
      %cmp = arith.cmpf ogt, %x, %zero : f32
      %y = arith.select %cmp, %x, %zero : f32
      memref.store %y, %X[%i] : memref<16xf32>
    }

    return
  }
}
