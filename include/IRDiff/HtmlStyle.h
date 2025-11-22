#pragma once
#include <string_view>

namespace irdiff {

inline constexpr std::string_view stringViewTemplate = R"(
<style>
body {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0b0b0e;
    color: #f0f0f0;
    margin: 0;
    padding: 1rem 1.5rem 3rem;
}
h1 {
    font-size: 1.4rem;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #aaaaaa;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}
.section {
    margin-top: 1.5rem;
}
.section-header {
    background: #181822;
    color: #9cd2ff;
    padding: 0.4rem 0.6rem;
    font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.9rem;
    border-radius: 4px 4px 0 0;
    border: 1px solid #27273a;
    border-bottom: none;
}
.diff-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.8rem;
    border: 1px solid #27273a;
    border-radius: 0 0 4px 4px;
    overflow: hidden;
}
.diff-table thead {
    background: #151521;
}
.diff-table th, .diff-table td {
    padding: 2px 6px;
    vertical-align: top;
    white-space: pre;
    overflow: hidden;
    text-overflow: ellipsis;
}
.diff-table th {
    border-bottom: 1px solid #27273a;
}
.diff-table .ln {
    width: 3rem;
    color: #888;
    text-align: right;
    border-right: 1px solid #27273a;
    background: #111119;
}
.diff-table .code {
    width: 47%;
}
tr.equal td.code {
    background: #101018;
}
tr.delete td.code:first-of-type {
    background: #3a1515;
}
tr.insert td.code:last-of-type {
    background: #123818;
}
tr.replace td.code {
    background: #3a3515;
}
tr.delete td.code:first-of-type,
tr.insert td.code:last-of-type,
tr.replace td.code {
    border-left: 2px solid #ff5555;
}
tr.insert td.code:last-of-type {
    border-left-color: #33dd66;
}
tr.replace td.code {
    border-left-color: #e0cc55;
}
</style>
)";

} // namespace irdiff