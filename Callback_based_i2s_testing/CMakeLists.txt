cmake_minimum_required(VERSION 3.10)
project(CLDL LANGUAGES CXX CUDA)

find_package(JetsonGPIO)



add_executable(tests
        Testing_Callback.cpp
        )

target_link_libraries(tests JetsonGPIO::JetsonGPIO)