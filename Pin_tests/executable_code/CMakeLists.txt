cmake_minimum_required(VERSION 3.10)

project(tests LANGUAGES CXX C)

add_executable(tests
    Initial_Testing.cpp)


set(tests_force_shared_crt ON FORCE)

target_link_libraries(tests PRIVATE Pins)

