#include "IRDiff/HtmlRender.h"
#include "IRDiff/ParseIR.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " old.ir new.ir [output.html]\n";
    return 1;
  }

  std::string oldPath    = argv[1];
  std::string newPath    = argv[2];
  std::string outputPath = (argc == 4) ? argv[3] : "ir_diff.html";

  try {
    auto oldSections = irdiff::parseIR(oldPath);
    auto newSections = irdiff::parseIR(newPath);

    std::string title       = "IR Diff: " + oldPath + " vs " + newPath;
    std::string htmlContent = irdiff::renderFullHtml(oldSections, newSections, title);

    std::ofstream outFile(outputPath);
    if (!outFile)
      throw std::runtime_error("Failed to open output file: " + outputPath);

    outFile << htmlContent;
    std::cout << "Diff HTML generated at: " << outputPath << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}