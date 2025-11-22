/// Util.h: Shared utility functions for trimming and HTML escaping.
#pragma once

#include <cctype>
#include <string>

namespace irdiff {
/// @brief Trims leading and trailing whitespace from a string.
/// @param str The input string to trim.
/// @return A new string with leading and trailing whitespace removed.
inline std::string trim(const std::string &str) {
  std::size_t start = 0;
  while (start < str.size() && std::isspace(static_cast<unsigned char>(str[start])))
    ++start;
  std::size_t end = str.size();
  while (end > start && std::isspace(static_cast<unsigned char>(str[end - 1])))
    --end;
  return str.substr(start, end - start);
}

/// @brief Escapes special HTML characters in a string.
/// @param str The input string to escape.
/// @return A new string with special HTML characters replaced by their escape
/// sequences.
inline std::string htmlEscape(const std::string &str) {
  std::string escaped;
  escaped.reserve(str.size());
  for (std::string::const_iterator it = str.begin(); it != str.end();
       ++it) { // Explicit iterator to avoid range-based for loop.
    char c = *it;
    switch (c) {
    case '&':
      escaped += "&amp;";
      break;
    case '<':
      escaped += "&lt;";
      break;
    case '>':
      escaped += "&gt;";
      break;
    case '"':
      escaped += "&quot;";
      break;
    case '\'':
      escaped += "&#39;";
      break;
    default:
      escaped.push_back(c);
      break;
    }
  }
  return escaped;
}
} // namespace irdiff