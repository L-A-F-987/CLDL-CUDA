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
Note that this command requires that the python file requires that the board only has an Sd card or emmc, it will not work if you have a custom board containing both and will result in an error saying multiple partitions are mounted. As such you'll need to work wth the pinmux spreadsheet provided at this link (https://developer.nvidia.com/embedded/downloads). Instructions to follow this method can be found here (https://forums.developer.nvidia.com/t/how-to-use-the-jetson-nanos-pinmux-spreadsheet/76221)

- /opt/nvidia/jetson-io/jetson-io.py

After running this command you will be met with the below screen. Note that your terminal window has to be big enough to display the menu so if your menu does not pop up try resizing your terminal window first.

![alt text](https://global.discourse-cdn.com/nvidia/original/3X/b/5/b5d719c1726d25a99b96c4582647296e9bc0fe6a.png)

On this window select Configure Manually and navigate to the i2s option. Press enter on this option and a "star"/asterix should appear next to it, then select save pin changes followed by reboot and configure new pins changes.

After this has been done, the pins should be configured for I2S as follows.

- Pin 12 = I2S SCLK or the clock frequency
  
- Pin 35 = I2S Fs or Sampling frequency (note this is also identical to word select if you are following the names from the actual I2S documentation)
  
- Pin 38 = I2S Data In

- Pin 40 = I2S Data Out

# Troubleshooting I2S
Before moving on please run the below command to check for a potential issue identified by this forum post (https://forums.developer.nvidia.com/t/enabling-i2s-audio-on-jetson-nano-gpio-header/245651). 

  
- sudo cat /sys/kernel/debug/tegra_gpio

After running check if the line J: ........ has the values J: 2:1 f0 00 00 00 00 00 000000, if so you need to change the f0 to be 00 to do this follow the below instructions. If it says 00 in the place of f0 you can skip the following debug section.

## Debugging tegra_gpio

In order to rectify this issue you have to edit the bootcmd, to do this you will require a serial debugger cable like the one shown below.

![image of serial Debug cable
](https://uk.pi-supply.com/cdn/shop/products/26849615c765531f0b2b74b598b70550_1200x901.jpg?v=1571708693)

Plug this serial cable into your jetson following this video (https://jetsonhacks.com/2019/04/19/jetson-nano-serial-console/), ensure that power is not connected when doing this. 

After attaching the cable to the jetson and your laptop, open a serial terminal on your laptop, I did this using the screen command on Mac following this example (https://forums.developer.nvidia.com/t/how-exactly-do-i-change-bootcmd-as-in-another-topic/298039). Run ls /dev/ on your local machine, you should see a tty.usbserial followed by some numbers, connect to this using screen on Mac (screen /dev/the_tty.usbserial_you_identified 115200) or some other terminal command on windows/linux. 

Plug in the jetson into power, you should then see a series of lines of code representing the boot appear on your terminal. When prompted interupt the bootloaded by pressing a key, after interupting this run the below commands.

- setenv bootcmd 'mw 0x6000d204 0; mw 0x6000d60c 0; run distro_bootcmd'
- saveenv
- reset

The file should now have 00 in place of f0.

# Default asound commands

## Soundcards
The jetson comes preconfigured with multiple soundcards (note you may also have extra due to external connecters such as the USB -> Stereo adapter), you can check which cards you have by running the below command. 

- cat /proc/asound/cards 

The default cards for the Nano Developer are provided below, you can find the default cards for other models at this link (https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3275/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/asoc_driver.19.2.html#wwpID0E0QX0HA). 

- tegrasndt210ref (AHUB card)
- tegrahda (HDA)

## Recording a Wav File 

To record a wav file to test the microphones you can run the below command found at the same link as the soundcards for other models. Fill in the spaces with acceptable settings, the second command shown is filled in, it will create a wav file with 2 channels, a sampling rate of 44100Hz with 16bit resolution called example.wav

- arecord -D hw:<cardname>,<i-1> -r rate -c channels -f sample_format out.wav
- arecord -D hw:tegrasndt210ref,0 -r 44100 -c 2 -f S16 example.wav

### Issues with arecord 

If you are having problems getting your microphone to work (in my case no error but the wav file is blank/empty), it may be worthwhile to try running the arecord command and varifying the clock signal and word select signals are working when not connected to the microphone by using an osilliscope, you can see how they should look in the below. In my case, I resolved this issue

![Example](https://github.com/user-attachments/assets/71ee7dd5-04a2-49fc-8dd3-5b7501fd6c6b)

Board output for I2S clock and I2S_Fs (1v per division)

![IMG_3916](https://github.com/user-attachments/assets/24d0230c-adf1-40f9-9c1f-940fc1be5e17)

![IMG_3917](https://github.com/user-attachments/assets/48364c60-e3e6-4d6b-a799-11ab3591fb21)

# Audio Playback

If like me, you would like to play audio back through a usb -> stereo adapter, you should plug in the usb adapter and then re-run the command to view all audio card, shown again below. It should then give an output similar to the one I have shown below with the usb card name and number being provided.

- cat /proc/asound/cards 

<img width="557" alt="image" src="https://github.com/user-attachments/assets/e5be9778-61db-4848-8f4d-f0941952f0f3">














  
 

## Contact

Luca Faccenda: 2572705f@student.glasgow.ac.uk
