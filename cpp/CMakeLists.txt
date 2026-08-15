cmake_minimum_required(VERSION 3.16)

project(rational_elliptic_phase_a LANGUAGES CXX)

add_executable(phase_a phase_a.cpp)
target_compile_features(phase_a PRIVATE cxx_std_17)

