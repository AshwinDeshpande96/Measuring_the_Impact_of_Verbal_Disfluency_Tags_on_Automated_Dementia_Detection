# How You Say It Matters: Measuring the Impact of Verbal Disfluency Tags on Automated Dementia Detection

This repository contains code developed for the paper entitled "How You Say It Matters: Measuring the Impact of Verbal Disfluency Tags on Automated Dementia Detection"

NOTE ON DATA: While the data used in this paper are publicly available, we are not able to redistribute any of these data as per Data Use agreement with Dementia Bank and the Carolinas Conversations Collection. In order to obtain these data, individual investigators need to contact the Dementia Bank and CCC and request access to the data.

## Data and Task Setup

We use ADReSS data published as a part of INTERSPEECH 2020 challenge. This data consists of spontaneous speech data for two groups:
* HD - Control - Subjects with no cognitive decline
* AD - Dementia - Subjects with mild cognitive impairment

The data consists of audio clips of roughly <average_audio_length> seconds long for <number_of_participants> speakers, along with manually transcribed (According to CHAT protocol) text.
Each clip consist of utterances belong to one of two participants: the subject and interviewer.

Following steps are carried out to prepare the data:
1. Prepare Gold and ASR Utterances:[[Code]](./code/speech_recognition/transcribe.py)
   * Identify subject utterances mentioned in manually transcribed text along with it's segment in audio 
      * ex: an(d) &uh the window's open . 13310_20608 - 13310ms to 20608ms in audio clip
   * Extract audio segment for audio which consists of subject's speech
   * Transcribe each segment using Google ASR
2. Remove disfluencies from manual transcripts:[[Code]](./code/disfluency/remove_disfluency_gold.py)
   * We assess the effect of disfluency as a predictor in dementia detection by selectively removing disfluencies of three types:
      * Repetition - Word or Phrase Repetition - Identified by '[/]' token after the reparandum
         * ex: <what are> [/] what are the instructions ? [+ exc] 19000_20484 - <what are> phrase is repeated
      * Retracing - Word or Phrase Retracing
         * ex: and there are dishes [//] &uh &uh two cups and a saucer on the sink. 37402_42053 - dishes is changed to two cups and a saucer
      * Both
4. Tag disfluency in ASR utterance [[Code]](https://github.com/pariajm/joint-disfluency-detector-and-parser)
5. Remove disfluencies from ASR transcripts:[[Code]](./code/disfluency/remove_disfluency_fisher.py)


