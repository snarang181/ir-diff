/// IRTypes.h: Core IR data structures and types for IRDiff.
#pragma once

#include <string>
#include <vector>

namespace irdiff {

struct OpLine {
  std::string text;
  int         lineNo; // Line number in the original IR file
};

struct Section {
  std::string         header;     // Section header text e.g. "func @foo(...)"
  std::vector<OpLine> operations; // List of operations in this section
};
} // namespace irdiff