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
Note that this command requires that the python file requires that the board only has an Sd card or emmc, it will not work if you have a custom board containing both and will result in an error saying multiple partitions are mounted. In order to 

- /opt/nvidia/jetson-io/jetson-io.py

 

## Contact
Luca Faccenda: 2572705f@student.glasgow.ac.uk
