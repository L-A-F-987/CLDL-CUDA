#include <iostream>
#include <chrono>
#include <thread>
#include <functional>


//Setting general variables 
const int  sampling_rate = 44100;
const int bit_resolution = 24;

//Stereo or mono 
//1 for mono two for Stereo 
const int number_of_microphones = 1;



const int Hz_for_bits = (sampling_rate*bit_resolution*number_of_microphones);


int main(){
    std::cout<<Hz_for_bits<<"\n\n\n\n\n";
    return 0;
}




