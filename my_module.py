import spacy


# return a list of lists containing the tokens.
def clean_text(file_path: str) -> list:
    text = open(file_path)
    sentence_lists = list()
    for line in text:
        sentence = line.strip()
        sub_sentences = sentence.split(".")
        for sub_sent in sub_sentences:
            # remove empty lines or lines with white space
            if not sub_sent.strip():
                continue
            sentence_lists.append(sub_sent.strip() + ".")
    return sentence_lists
