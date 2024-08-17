#include <alsa/asoundlib.h>
#include <stdio.h>
#include <string.h>


static char *device = "hw:2,0";            /* playback device */
unsigned char buffer[16*1024*2];              /* some random data */

int main(){

printf("\n\n\nMade it to the start :)\n\n\n");



int err;
    unsigned int i;
    snd_pcm_t *handle;
    snd_pcm_sframes_t frames;
 
    for (i = 0; i < sizeof(buffer); i++)
        buffer[i] = random() & 0xff;
 
    if ((err = snd_pcm_open(&handle, device, SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
        printf("Playback open error: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }
    if ((err = snd_pcm_set_params(handle,
                      SND_PCM_FORMAT_S16_LE,
                      SND_PCM_ACCESS_RW_INTERLEAVED,
                      2,
                      48000,
                      1,
                      500000)) < 0) {   /* 0.5sec */
        printf("Playback open error: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }
 
        for (i = 0; i < 16; i++) {
        frames = snd_pcm_writei(handle, buffer, sizeof(buffer));
        if (frames < 0)
            frames = snd_pcm_recover(handle, frames, 0);
        if (frames < 0) {
            printf("snd_pcm_writei failed: %s\n", snd_strerror(frames));
            break;
        }
        if (frames > 0 && frames < (long)sizeof(buffer))
            printf("Short write (expected %li, wrote %li)\n", (long)sizeof(buffer), frames);
    }
 
    /* pass the remaining samples, otherwise they're dropped in close */
    err = snd_pcm_drain(handle);
    if (err < 0)
        printf("snd_pcm_drain failed: %s\n", snd_strerror(err));
    snd_pcm_close(handle);
    return 0;
}