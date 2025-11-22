/// ParseIR.cpp: Implementation of the IR parsing functions.
#include "IRDiff/ParseIR.h"
#include "IRDiff/Util.h"

#include <fstream>
#include <stdexcept>

namespace irdiff {
std::vector<Section> parseIR(const std::string &path) {
  std::ifstream inFile(path);
  if (!inFile)
    throw std::runtime_error("Failed to open IR file: " + path);

  std::vector<Section> sections;
  std::string          currentHeader = "(module)";
  std::vector<OpLine>  currentOps;

  std::string line;
  int         lineNo = 0;

  while (std::getline(inFile, line)) {
    ++lineNo;
    std::string trimmedLine = trim(line);

    // Skip empty and comment lines
    if (trimmedLine.empty() || trimmedLine.rfind("//", 0) == 0)
      continue;

    bool isFunc = (trimmedLine.rfind("func ", 0) == 0) || (trimmedLine.rfind("func @", 0) == 0) ||
                  (trimmedLine.rfind("llvm.func", 0) == 0);
    if (isFunc) {
      if (!currentOps.empty()) {
        sections.push_back(Section{currentHeader, currentOps});
        currentOps.clear(); // Reset for new section
      }
      currentHeader = trimmedLine; // New section header
    } else {
      // Regular operation line
      currentOps.push_back(OpLine{trimmedLine, lineNo});
    }
  }
  // Add the last section if it has operations
  if (!currentOps.empty()) {
    sections.push_back(Section{currentHeader, currentOps});
  }
  return sections;
}
} // namespace irdiff