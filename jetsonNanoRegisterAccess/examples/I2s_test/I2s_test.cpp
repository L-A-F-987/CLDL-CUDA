#include <iostream>
#include <string>
#include <cassert>
#include <unistd.h>

#include "../../peripheralController/peripheralController.h"
#include "../../gpioController/gpio.h"
#include "../../pinmuxController/pinmuxController.h"


//Force J41-pin38 to input mode


int main()
{	
    
    PeripheralController myGpioController1(gpioController::gpioController1BaseAddress);
	PeripheralController myGpioController2(gpioController::gpioController2BaseAddress);
	PeripheralController myGpioController3(gpioController::gpioController3BaseAddress);
	PeripheralController myGpioController4(gpioController::gpioController4BaseAddress);
	PeripheralController myGpioController5(gpioController::gpioController5BaseAddress);
	PeripheralController myGpioController6(gpioController::gpioController6BaseAddress);
	PeripheralController myGpioController7(gpioController::gpioController7BaseAddress);
	PeripheralController myGpioController8(gpioController::gpioController8BaseAddress);
    PeripheralController myPinMuxController(pinmuxController::baseAddress);
    
    uint32_t gpioPinState = gpioController::BIT_N_HIGH;

    //setting up pin 38 as an input
    myGpioController3.setRegisterField(GPIO_CNF_1_RMW::addressOffset, gpioController::BIT_N_GPIO, GPIO_CNF_1_RMW::BIT_5_baseBit, GPIO_CNF_1_RMW::BIT_5_bitWidth);
    myPinMuxController.setRegisterField(PINMUX_AUX_DAP4_DIN_0::addressOffset, pinmuxController::PUPD_BIT_PULL_UP, PINMUX_AUX_DAP4_DIN_0::PUPD_bit, PINMUX_AUX_DAP4_DIN_0::PUPD_bitWidth);
    myPinMuxController.setRegisterField(PINMUX_AUX_DAP4_DIN_0::addressOffset, pinmuxController::E_INPUT_BIT_ENABLE, PINMUX_AUX_DAP4_DIN_0::E_INPUT_bit, PINMUX_AUX_DAP4_DIN_0::E_INPUT_bitWidth);

    uint32_t readGPIOINPUT = 0;


while(true)
    {
        sleep(1);
        readGPIOINPUT = myGpioController3.getRegisterField(GPIO_IN_1_RMW::addressOffset, GPIO_IN_1_RMW::BIT_5_baseBit, GPIO_IN_1_RMW::BIT_5_bitWidth);
		std::cout<<"pin 38 INPUT bit: "<<readGPIOINPUT<<std::endl;

    	}
    
return 0;
}