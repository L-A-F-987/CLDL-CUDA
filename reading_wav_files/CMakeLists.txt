cmake_minimum_required(VERSION 3.10)
# 'Google_test' is the subproject name
project(tests LANGUAGES CXX C)



add_executable(wav_info
        Reading_wav_file.cpp
        )

        