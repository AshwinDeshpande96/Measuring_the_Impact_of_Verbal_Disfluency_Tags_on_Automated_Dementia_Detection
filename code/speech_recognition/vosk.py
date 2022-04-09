#!/usr/bin/env python3
from jiwer import wer
from vosk import Model, KaldiRecognizer, SetLogLevel
import enchant
import os
import wave
from ast import literal_eval

# initialisation
d = enchant.Dict("en_US")


def get_vocab(vocabfile):
    vocab = ""
    text = open(vocabfile, 'r')
    for word in text.read().split('\n'):
        if d.check(word):
            vocab += word + " "
    return vocab.lower()


vocab = get_vocab('./vocab')
SetLogLevel(0)

if not os.path.exists("model"):
    print(
        "Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit(1)


def asr(wf, rec):
    while True:
        data = wf.readframes(8000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = (rec.Result())
        else:
            res = (rec.PartialResult())
    return literal_eval(rec.FinalResult())


target_transcript = {f.split()[0]: " ".join(f.split()[1:]) for f in sorted(open('text_train').read().split('\n'))}
result_transcript = {}
audio_file = {f.split()[0]: " ".join(f.split()[1:]) for f in sorted(open('wav_train.scp').read().split('\n'))}
model = Model("model2")
# model = Model("model")  # Uncomment to use Aspire Model
i = 0
avg_wer = 0
for key, file in audio_file.items():
    spkr = key.split('_')[0]
    if spkr not in result_transcript:
        result_transcript[spkr] = []
    try:
        wf = wave.open(file, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio file must be WAV format mono PCM.")
            exit(1)
        # Using ADReSSo vocab.
        # rec = KaldiRecognizer(model, wf.getframerate(), '["{}", "[unk]"]'.format(vocab)) #Uncomment to use

        # Using Basic Vosk
        rec = KaldiRecognizer(model, wf.getframerate())
        result = asr(wf, rec)['text']
        target = target_transcript[key]
        error = wer(target.split(), result.split())
        print("Target: ", target)
        print("Result: ", result)
        if result:
            result_transcript[spkr].append(result)
        print("WER: ", error)
        avg_wer += error
        i += 1
        print("Cumulative avg:", avg_wer / i)
        print()
    except Exception as e:
        pass

# output_fptr = open('output_text_test.csv', 'a')
# for spkr, utterances in result_transcript.items():
#     utterances = " . ".join(utterances)
#     line = "{},{}\n".format(spkr, utterances)
#     output_fptr.write(line)
# output_fptr.close()
