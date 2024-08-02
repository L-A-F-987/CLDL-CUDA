cmake_minimum_required(VERSION 3.10)
project(Pins LANGUAGES CXX C)

add_subdirectory(executable_code)

enable_testing()


add_library(Pins STATIC
        jetsonGPIO.c)

  set_target_properties(Pins PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  PUBLIC_HEADER "${LIBINCLUDE}")

  target_include_directories(Pins
  PUBLIC
  include/)

  find_package(Threads)

target_link_libraries(Pins curand ${CMAKE_THREAD_LIBS_INIT})


