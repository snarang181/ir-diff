/// Diff.h: Diffing functionality for IRDiff -- enums/APIs.
#pragma once

#include "IRDiff/IRTypes.h"
#include <string>
#include <vector>

namespace irdiff {
enum class Tag {
  Equal,  // No change
  Delete, // Present in old IR, absent in new IR
  Insert, // Absent in old IR, present in new IR
  Replace // Present in both, but different
};

struct DiffRow {
  Tag         tag;       // Type of difference
  std::string leftText;  // Text from the old IR
  std::string rightText; // Text from the new IR
  int         leftLine;  // -1 if not applicable
  int         rightLine; // -1 if not applicable
};

/// @brief Diff two lists of OpLines and produce a list of DiffRows.
/// @param oldOps Operations from the old IR section.
/// @param newOps Operations from the new IR section.
/// @return A vector of DiffRows representing the differences.
std::vector<DiffRow> diffOps(const std::vector<OpLine> &oldOps, const std::vector<OpLine> &newOps);
} // namespace irdiff
