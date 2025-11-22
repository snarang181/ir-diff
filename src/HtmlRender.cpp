#include "IRDiff/HtmlRender.h"
#include "IRDiff/Diff.h"
#include "IRDiff/HtmlStyle.h"
#include "IRDiff/Util.h"

#include <sstream>
#include <string_view>
#include <unordered_map>

namespace irdiff {

static void renderSectionHtml(std::ostream &out, const std::string &header, const Section *oldSec,
                              const Section *newSec) {
  const std::vector<OpLine> emptyOps;
  const auto               &oldOps = oldSec ? oldSec->operations : emptyOps;
  const auto               &newOps = newSec ? newSec->operations : emptyOps;

  auto rows = diffOps(oldOps, newOps);

  out << "<div class=\"section\">\n";
  out << "<div class=\"section-header\">" << htmlEscape(header) << "</div>\n";
  out << "<table class=\"diff-table\">\n";
  out << "<thead><tr>"
      << "<th class='ln'>#</th><th>Old</th>"
      << "<th class='ln'>#</th><th>New</th>"
      << "</tr></thead><tbody>\n";

  for (const auto &row : rows) {
    std::string rowClass;
    switch (row.tag) {
    case Tag::Equal:
      rowClass = "equal";
      break;
    case Tag::Delete:
      rowClass = "delete";
      break;
    case Tag::Insert:
      rowClass = "insert";
      break;
    case Tag::Replace:
      rowClass = "replace";
      break;
    }

    std::string leftLine  = row.leftLine != -1 ? std::to_string(row.leftLine) : "";
    std::string rightLine = row.rightLine != -1 ? std::to_string(row.rightLine) : "";

    out << "<tr class='" << rowClass << "'>";
    out << "<td class='ln'>" << htmlEscape(leftLine) << "</td>";
    out << "<td class='code'>" << htmlEscape(row.leftText) << "</td>";
    out << "<td class='ln'>" << htmlEscape(rightLine) << "</td>";
    out << "<td class='code'>" << htmlEscape(row.rightText) << "</td>";
    out << "</tr>\n";
  }
  out << "</tbody></table>\n</div>\n";
}

std::string renderFullHtml(const std::vector<Section> &oldSections,
                           const std::vector<Section> &newSections, const std::string &title) {
  std::ostringstream html;

  std::unordered_map<std::string, const Section *> oldSectionMap;
  std::unordered_map<std::string, const Section *> newSectionMap;
  for (const auto &sec : oldSections)
    oldSectionMap[sec.header] = &sec;
  for (const auto &sec : newSections)
    newSectionMap[sec.header] = &sec;

  std::vector<std::string> allHeaders;
  allHeaders.reserve(oldSectionMap.size() + newSectionMap.size());
  for (const auto &pair : oldSections)
    allHeaders.push_back(pair.header);
  for (const auto &pair : newSections) {
    // Avoid duplicates
    if (oldSectionMap.find(pair.header) == oldSectionMap.end())
      allHeaders.push_back(pair.header);
  }
  // HTML Header
  html << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta "
          "charset=\"utf-8\" />\n";
  html << "<title>" << htmlEscape(title) << "</title>\n";
  html << stringViewTemplate;
  html << "</head>\n<body>\n";
  html << "<h1>" << htmlEscape(title) << "</h1>\n";
  html << "<div class=\"subtitle\">IR Diff Viewer</div>\n";

  // Render each section
  for (const auto &header : allHeaders) {
    const Section *oldSec = nullptr;
    const Section *newSec = nullptr;
    auto           oldIt  = oldSectionMap.find(header);
    if (oldIt != oldSectionMap.end())
      oldSec = oldIt->second;
    auto newIt = newSectionMap.find(header);
    if (newIt != newSectionMap.end())
      newSec = newIt->second;
    renderSectionHtml(html, header, oldSec, newSec);
  }

  // HTML Footer
  html << "\n</body>\n</html>\n";
  return html.str();
}
} // namespace irdiff