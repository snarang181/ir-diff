// new.mlir

module {
  // Matmul lowered to linalg + slightly different constants
  func.func @matmul_naive(%A: memref<4x4xf32>,
                     %B: memref<4x4xf32>,
                     %C: memref<4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1.0 : f32

    // Initialize C to 1.0 instead of 0.0
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        memref.store %one, %C[%i, %j] : memref<4x4xf32>
      }
    }

    // Use linalg.matmul instead of explicit triple loop
    %viewA = memref.cast %A : memref<4x4xf32> to memref<4x4xf32>
    %viewB = memref.cast %B : memref<4x4xf32> to memref<4x4xf32>
    %viewC = memref.cast %C : memref<4x4xf32> to memref<4x4xf32>

    linalg.matmul ins(%viewA, %viewB : memref<4x4xf32>, memref<4x4xf32>)
                  outs(%viewC : memref<4x4xf32>)

    return
  }

  // ReLU fused with bias add, plus changed signature
  func.func @relu(%X: memref<16xf32>, %bias: f32) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32

    scf.for %i = %c0 to %c16 step %c1 {
      %x = memref.load %X[%i] : memref<16xf32>
      %xb = arith.addf %x, %bias : f32
      %cmp = arith.cmpf ogt, %xb, %zero : f32
      %y = arith.select %cmp, %xb, %zero : f32
      memref.store %y, %X[%i] : memref<16xf32>
    }

    return
  }

  // New function only in the new file
  func.func @scale(%X: memref<16xf32>, %alpha: f32) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c16 step %c1 {
      %x = memref.load %X[%i] : memref<16xf32>
      %y = arith.mulf %x, %alpha : f32
      memref.store %y, %X[%i] : memref<16xf32>
    }

    return
  }
}
