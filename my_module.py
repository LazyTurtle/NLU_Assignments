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


def extract_path_to_token(doc, token) -> list:
    head_of = dict()
    for tok in doc:
        head_of[tok] = tok.head

    root = find_root(doc)
    head_of[root] = None

    temp = token
    path = list()

    while temp is not None:
        path.append(temp)
        temp = head_of[temp]
    path.reverse()
    return path


def find_root(doc) -> str:
    for token in doc:
        if token.dep_ == "ROOT":
            return token
