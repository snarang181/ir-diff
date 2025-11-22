#include "IRDiff/Diff.h"

#include <algorithm>
#include <vector>

namespace irdiff {
struct BackCell {
  Tag tag;
  int i; // Index in oldOps
  int j; // Index in newOps
};

std::vector<DiffRow> diffOps(const std::vector<OpLine> &oldOps, const std::vector<OpLine> &newOps) {
  const int oldSize = static_cast<int>(oldOps.size());
  const int newSize = static_cast<int>(newOps.size());

  std::vector<std::string> oldTexts, newTexts;
  oldTexts.reserve(oldSize);
  newTexts.reserve(newSize);
  for (const auto &op : oldOps)
    oldTexts.push_back(op.text);
  for (const auto &op : newOps)
    newTexts.push_back(op.text);

  // DP Table for Longest Common Subsequence (LCS)
  std::vector<std::vector<int>> dp(oldSize + 1, std::vector<int>(newSize + 1, 0));
  for (int i = 1; i <= oldSize; ++i) {
    for (int j = 1; j <= newSize; ++j) {
      if (oldTexts[i - 1] == newTexts[j - 1]) { // Match
        dp[i][j] = dp[i - 1][j - 1] + 1;        // Increment LCS length
      } else {
        dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]); // Take max from left or top
      }
    }
  }

  // Backtrack to fid differences
  std::vector<BackCell> backtrack;
  backtrack.reserve(oldSize + newSize);

  int i = oldSize;
  int j = newSize;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && oldTexts[i - 1] == newTexts[j - 1]) { // Match
      backtrack.push_back(BackCell{Tag::Equal, i - 1, j - 1});
      --i;
      --j;
    } else if (i > 0 && (j == 0 || dp[i - 1][j] >= dp[i][j - 1])) { // Delete
      backtrack.push_back(BackCell{Tag::Delete, i - 1, -1});
      --i;
    } else if (j > 0) { // Insert
      backtrack.push_back(BackCell{Tag::Insert, -1, j - 1});
      --j;
    }
  }
  std::reverse(backtrack.begin(),
               backtrack.end()); // Reverse to get correct order

  // Construct DiffRows from backtrack
  std::vector<DiffRow> diffs;
  diffs.reserve(backtrack.size());

  for (const auto &cell : backtrack) {
    switch (cell.tag) {
    case Tag::Equal: {
      const auto &oldOp = oldOps[cell.i];
      const auto &newOp = newOps[cell.j];
      diffs.push_back(DiffRow{Tag::Equal, oldOp.text, newOp.text, oldOp.lineNo, newOp.lineNo});
      break;
    }
    case Tag::Delete: {
      const auto &oldOp = oldOps[cell.i];
      diffs.push_back(DiffRow{Tag::Delete, oldOp.text, std::string(), oldOp.lineNo, -1});
      break;
    }
    case Tag::Insert: {
      const auto &newOp = newOps[cell.j];
      diffs.push_back(DiffRow{Tag::Insert, std::string(), newOp.text, -1, newOp.lineNo});
      break;
    }
    default:
      break;
    }
  }

  // Merge consecutive Deletes and Inserts into Replaces
  std::vector<DiffRow> mergedDiffs;
  for (std::size_t k = 0; k < diffs.size(); ++k) {
    const auto &current = diffs[k];
    if (current.tag == Tag::Delete && k + 1 < diffs.size() && diffs[k + 1].tag == Tag::Insert) {
      const auto &next = diffs[k + 1];
      mergedDiffs.push_back(DiffRow{Tag::Replace, current.leftText, next.rightText,
                                    current.leftLine, next.rightLine});
      ++k; // Skip the next as it's merged
    } else if (current.tag == Tag::Insert && k + 1 < diffs.size() &&
               diffs[k + 1].tag == Tag::Delete) {
      const auto &next = diffs[k + 1];
      mergedDiffs.push_back(DiffRow{Tag::Replace, next.leftText, current.rightText, next.leftLine,
                                    current.rightLine});
      ++k; // Skip the next as it's merged
    } else {
      mergedDiffs.push_back(current);
    }
  }
  return mergedDiffs;
}
} // namespace irdiff