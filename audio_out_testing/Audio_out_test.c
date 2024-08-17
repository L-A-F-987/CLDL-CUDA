#include <alsa/asoundlib.h>
#include <stdio.h>
#include <string.h>


static char *output_device = "hw:2,0";            /* playback device */

int rc;


//Creating variables for the output device
//
//Creating a variable for storing the output devie parameters
snd_pcm_hw_params_t * hardware_parameters_of_output_device; 

//Creating handle for output_device
 snd_pcm_t *handle_output;



int main(){

printf("\n\n\nMade it to the start :)\n\n\n");
//
//
//


//Opening the Output Device

rc = snd_pcm_open(&handle_output,output_device ,
                    SND_PCM_STREAM_PLAYBACK, 0);
  if (rc < 0) {
    fprintf(stderr,
            "unable to open output pcm device: %s\n",
            snd_strerror(rc));
    exit(1);
  }

//Allocating Hardware Parameters to the output device, this does the same as maloc however it
//Automatically frees the data when the function 

snd_pcm_hw_params_alloca(&hardware_parameters_of_output_device);

//Filling the params with defualt values
snd_pcm_hw_params_any(handle_output, hardware_parameters_of_output_device);

//Setting up the output device to be in interleaved mode 
// this means that there are 2 channels i.e. stereo

snd_pcm_hw_params_set_access(handle_output, hardware_parameters_of_output_device,
                      SND_PCM_ACCESS_RW_INTERLEAVED);

//Setting up the output device to be to be 16 bit

snd_pcm_hw_params_set_format(handle_output,hardware_parameters_of_output_device,
                                SND_PCM_FORMAT_S16_LE);

//set up 2 channels for output device  (stereo )

snd_pcm_hw_params_set_channels(handle_output,hardware_parameters_of_output_device,
                                2);

//Setting the sampling rate of the output device to be 44100Khz

int sampling_rate = 44100;
int dir;

snd_pcm_hw_params_set_rate_near(handle_output,hardware_parameters_of_output_device,
                                &sampling_rate, &dir);


//Set period size in terms for frames for the output device
// the period size is the bits times the number of channels ???????

snd_pcm_uframes_t frames = 32;

snd_pcm_hw_params_set_period_size_near(handle_output,hardware_parameters_of_output_device,
                                &frames, &dir);


///Setting up buffer

snd_pcm_uframes_t buffer_frames = 2 * frames;


 snd_pcm_hw_params_set_buffer_size_near(handle_output, hardware_parameters_of_output_device,
                                &buffer_frames);


/* Send the parameters to the driver for output device */
  rc = snd_pcm_hw_params(handle_output, hardware_parameters_of_output_device);
  if (rc < 0) {
    fprintf(stderr,
            "unable to set hw parameters for output device: %s\n",
            snd_strerror(rc));
    exit(1);
  }

//
//
//
// Setting software parameters of output device

snd_pcm_sw_params_t *software_parameters_of_output_device;

//does the same as it did for hardware
snd_pcm_sw_params_alloca(&software_parameters_of_output_device);



rc = snd_pcm_sw_params_current(handle_output,software_parameters_of_output_device);

if (rc < 0) {
    printf("Unable to determine current swparams for output: %s\n", snd_strerror(rc));
    return rc;
  }




rc = snd_pcm_sw_params_set_start_threshold(handle_output,software_parameters_of_output_device,
                                        0x7fffffff);
if (rc < 0) {
    printf("Unable to set start threshold mode for output device: %s\n", snd_strerror(rc));
    return rc;
  }



//Opening Device
snd_pcm_open(&handle_output, output_device, SND_PCM_STREAM_PLAYBACK, 0);




//closing Device
snd_pcm_close(handle_output);


    

 
//
//
//
 printf("\n\n\nMade it to the End :)\n\n\n");
    return 0;
}