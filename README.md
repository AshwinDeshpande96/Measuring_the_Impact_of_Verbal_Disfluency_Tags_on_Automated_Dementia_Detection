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
1. Prepare Gold and ASR Utterances:
    * Identify subject utterances mentioned in manually transcribed text along with it's segment in audio
    * Extract audio segment for audio which consists of subject's speech
    * Transcribe each segment using Google ASR
2. Identify disfluencies in each utterance


