cmake_minimum_required(VERSION 3.15)
project(ir_diff LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Library with core functionality
add_library(irdiff 
    src/ParseIR.cpp
    src/Diff.cpp
    src/HtmlRender.cpp
)

target_include_directories(irdiff PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Executable for command line tool
add_executable(ir_diff_cli src/main.cpp)
target_link_libraries(ir_diff_cli PRIVATE irdiff)