cmake_minimum_required(VERSION 2.8)
project(colour NONE)

include(RezBuild)

file(GLOB_RECURSE all_files "colour/*")
rez_install_files(
    ${all_files}
    DESTINATION .
)


