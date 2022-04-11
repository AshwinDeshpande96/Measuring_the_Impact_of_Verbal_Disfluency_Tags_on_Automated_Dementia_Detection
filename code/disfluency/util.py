import re


def update(new, old):
    t = None
    if new is not None:
        t = new
    if old is not None:
        t = old
    return t


def count_words(text):
    if text is None:
        return
    text = text.split()
    return len(text)


def get_percentage(num_words, total_words):
    if total_words is None or total_words == 0:
        return -1
    if num_words is None:
        num_words = 0
    return (total_words - num_words) / total_words


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z' ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_percentage(num_words, total_words):
    if total_words is None or total_words == 0:
        return -1
    if num_words is None:
        num_words = 0
    return (total_words - num_words) / total_words


def resolve_repeats(text):
    text_tokens = text.split()
    idx = 0
    while idx < len(text_tokens):
        token = text_tokens[idx]
        if token != "[x":
            idx += 1
            continue
        repeat_count = int(re.sub(r"[^0-9]", "", text_tokens[idx + 1])) - 1
        repeats = [text_tokens[idx - 1]] * repeat_count
        if idx + 2 < len(text_tokens):
            text_tokens = text_tokens[:idx] + repeats + text_tokens[idx + 2:]
        else:
            text_tokens = text_tokens[:idx] + repeats
        idx += 1
    return " ".join(text_tokens)


def remove_tags(text):
    text = text.replace("*PAR:\t", '')
    if text.endswith('...'):
        text = text[-3:]
    text = text.replace("*PAR:", '')
    text = text.replace("(...)", '')
    text = text.replace("...", '')
    text = text.replace("(..)", '')
    text = text.replace("(.)", '')
    text = text.replace('[//]', '')
    text = text.replace('[/]', '')
    text = text.replace('‡', '')
    text = text.replace('xxx', '')
    text = text.replace('[=! sings]', '')
    text = re.sub(r"[&()<>]", '', text)
    matches = re.findall(r'\[[\:\*][a-zA-Z\:\_\'\-\@\s]+\]', text)
    for match in matches:
        text = text.replace(match, '')
    if re.findall(r"\d+\_\d+", text):
        text = re.split(r'[\s/+"][.?!][\s[]', text)[0]
    text = re.sub(r"\d+\_\d+]", "", text)
    text = re.sub(r"\+", "", text)
    text = re.sub(r"\"", "", text)
    if text.startswith('.'):
        text = text[1:]
    text = re.sub(r"[=:][a-z]+", "", text)
    if re.findall(r"\[x[0-9 ]+\]", text):
        text = resolve_repeats(text)
    text = text.replace('[]', '')
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text_tokens = text.split()
    text_tokens = [t if "@" not in t else "" for t in text_tokens]
    text = " ".join(text_tokens)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
