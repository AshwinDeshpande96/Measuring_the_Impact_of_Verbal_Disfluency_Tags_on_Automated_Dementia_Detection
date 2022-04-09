import re

from global_var import *

language = "task_language=en"
text_type = "is_text_type=plain"
output_format = "os_task_file_format=json"
smil_audio_file = "os_task_file_smil_audio_ref={}"
smil_txt_file = "os_task_file_smil_page_ref={}"


def align_using_aeneas(audio_file, transcript):
    file_id = audio_file.split('/')[-1].split('.')[0]
    text_file = os.path.join("data", "aeneas_files", "text", "{}.txt".format(file_id))
    out_file = os.path.join("data", "aeneas_files", "out", "{}.json".format(file_id))
    CONFIG = "|".join([language,
                       text_type,
                       output_format])
    # smil_audio_file.format(audio_file),
    # smil_txt_file.format(text_file)])
    words = transcript.split()
    plain_text = "\n".join(words)
    with open(text_file, 'w') as fptr:
        fptr.write(plain_text)

    command = 'python -m aeneas.tools.execute_task {} {} "{}" {}'.format(audio_file, text_file, CONFIG, out_file)
    print(command)
    os.system(command)


def reformat_text(transcript):
    transcript = transcript.lower()
    transcript = transcript.replace("'s", "is")
    transcript = transcript.replace("'m", "am")
    transcript = transcript.replace("'re", "are")
    transcript = transcript.replace("'ll", "will")
    transcript = transcript.replace("'ve", "have")
    transcript = transcript.replace("'d", "would")
    transcript = transcript.replace("_", " ")
    transcript = transcript.replace(":", " ")
    transcript = transcript.replace("+", " ")
    transcript = transcript.replace("n't", "not")
    transcript = re.sub("[^a-zA-Z ]", " ", transcript)
    transcript_tokens = list(tokenizer.tokenize(transcript))
    transcript = " ".join(transcript_tokens)
    transcript = re.sub("\#", " ", transcript)
    transcript = re.sub(r"\s+", " ", transcript).strip()
    return transcript


def align_using_penn(audio_file, transcript, transcript_type):
    transcript = reformat_text(transcript)
    if system_type == "windows":
        audio_file = audio_file.replace("/", '\\')
    file_id = audio_file.split(directory_seperator)[-1].split('.')[0]
    text_file = os.path.join("data", "penn_files", "text", f"{file_id}_{transcript_type}.txt")
    wout_file = os.path.join("data", "penn_files", "word_out", f"{file_id}_{transcript_type}.pickle")
    pout_file = os.path.join("data", "penn_files", "phoneme_out", f"{file_id}_{transcript_type}.pickle")
    with open(text_file, 'w') as fptr:
        fptr.write(transcript)
        fptr.close()
    try:
        phoneme_alignments, word_alignments = align.align(audio_file, text_file)
        pickle.dump(word_alignments, open(wout_file, 'wb'))
        pickle.dump(phoneme_alignments, open(pout_file, 'wb'))
    except Exception as e:
        print(file_id, e)
