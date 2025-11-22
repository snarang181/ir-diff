// HtmlRender.h": Public API for rendering diffs as HTML.
#pragma once

#include "IRDiff/IRTypes.h"
#include <string>
#include <vector>

namespace irdiff {
/// @brief Render the diff between two IR sections as an HTML string.
/// @param oldSection The section from the old IR.
/// @param newSection The section from the new IR.
/// @return An HTML string representing the diff.
std::string renderFullHtml(const std::vector<Section> &oldSections,
                           const std::vector<Section> &newSections, const std::string &title);
} // namespace irdiff