# Deep Neural Filter Hearing Aid

 The objective of this project was to design a real-time adaptive noise canceling hearing aid style device. Unlike conventional hearing aids, this hearing aid used a neural network (Deep neural filter) libary written by Dr Bernd Porr at University of Glasgow. 
 
# Prerequisites

 A CUDA-enabled GPU is required to use this library.
 
 The CUDA developer toolkit is required to compile and run the library.

 Tested with CUDA-10.2 on a Jetson Nano Dev Kit
``
# Hardware 
- Jetson Nano Developer Kit

- Two I2s compatable Microphones

- 3.5mm Stereo Headphones

- USB to 3.5mm Stereo 

# Setting up I2S
To setup I2S on the jetson first you should run the below command. 
Note that this command requires that the python file requires that the board only has an Sd card or emmc, it will not work if you have a custom board containing both and will result in an error saying multiple partitions are mounted. As such you'll need to work wth the pinmux spreadsheet provided at this link (https://developer.nvidia.com/embedded/downloads).

- /opt/nvidia/jetson-io/jetson-io.py

After running this command you will be met with the below screen. Note that your terminal window has to be big enough to display the menu so if your menu does not pop up try resizing your terminal window first.

![alt text]

On this window select Configure Manually and navigate to the i2s option. Press enter on this option and a "star"/asterix should appear next to it, then select save pin changes followed by reboot and configure new pins changes.

After this has been done, the pins should be configured for I2S as follows.

- Pin 12 = I2S SCLK or the clock frequency
  
- Pin 35 = I2S Fs or Sampling frequency (note this is also identical to word select if you are following the names from the actual I2S documentation)
  
- Pin 38 = I2S Data In

- Pin 40 = I2S Data Out

# Troubleshooting I2S
Before moving on please run the below command to check for a potential issue identified by this forum post (https://forums.developer.nvidia.com/t/enabling-i2s-audio-on-jetson-nano-gpio-header/245651). 

  
- sudo cat /sys/kernel/debug/tegra_gpio

After running check if the line J: ........ has the values J: 2:1 f0 00 00 00 00 00 000000, if so you need to changed the f0 to be 00 to do this follow the below instructions.

# Debugging tegra_gpio

In order to rectify this issue you have to edit the bootcmd, to do this you will require a serial debugger cable like the one shown below.

![image of serial Debug cable
](https://uk.pi-supply.com/cdn/shop/products/26849615c765531f0b2b74b598b70550_1200x901.jpg?v=1571708693)




  
 

## Contact
Luca Faccenda: 2572705f@student.glasgow.ac.uk
