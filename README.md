# IR Diff Visualizer

A small C++ tool for **side-by-side visual diffs of IR** (MLIR / LLVM-IR / Triton-IR style).

Given two IR files, it:

- Splits them into **sections** (functions detected via `func @...`, `func ...`, `llvm.func ...`)
- Computes a **line-level diff** for each section using an LCS-based algorithm
- Renders a dark-themed **HTML report** with a two-column view: “Old” vs “New”
- Highlights:
  - unchanged lines
  - deleted lines
  - inserted lines
  - replaced lines

## Building and Running
Requires a C++17-capable compiler and CMake.

```bash
git clone https://github.com/samarthnarang/ir-diff.git
cd ir-diff
mkdir build && cd build
cmake ..
cmake --build .
./ir-diff_cli <old_file.ir> <new_file.ir>  <output_report.html>
```

## Example Usage (with sample files)
```bash
# Simple case, default output name to ir_diff.html
./ir-diff_cli ../samples/old.mlir ../samples/new.mlir
# Custom output name
./ir-diff_cli ../samples/old.mlir ../samples/new.mlir ../samples/diff.html
```
