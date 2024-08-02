// exampleApp.c

#include <iostream>
#include <string>
#include <unistd.h>
#include "jetsonGPIO.h"

using namespace std;

int main(int argc, char *argv[]){

    cout << "Testing the GPIO Pins"<<endl;
    jetsonGPIO redLED = gpio165 ;
    cout << redLED<<endl;

    for(int i=0; i<5; i++){
        cout << "Setting the LED on" << endl;
        gpioSetValue(redLED, on);
        usleep(200000);         // on for 200ms
        cout << "Setting the LED off" << endl;
        gpioSetValue(redLED, off);
        usleep(200000);         // off for 200ms
    }
    

    return 0;
}