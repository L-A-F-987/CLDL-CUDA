cmake_minimum_required(VERSION 3.10)
# 'Google_test' is the subproject name
project(tests LANGUAGES CXX C)



add_executable(audio_out_tests
        Audio_out_test.c
        )

        
find_package(ALSA REQUIRED)                                    
if (ALSA_FOUND)                                                
    include_directories(${ALSA_INCLUDE_DIRS})                  
    target_link_libraries (audio_out_tests ${ALSA_LIBRARIES})      
endif(ALSA_FOUND)