/// ParseIR.h: Public API for parsing IR.
#pragma once

#include "IRDiff/IRTypes.h"
#include <string>
#include <vector>

namespace irdiff {

/// @brief Parse a textual IR file into a list of Sections.
/// @details
/// Very simple heuristic-based parser:
/// - Lines starting with a known section header keyword (e.g., "func", "global")
/// - skip empty lines
/// @param path
/// @return
std::vector<Section> parseIR(const std::string &path);
} // namespace irdiff