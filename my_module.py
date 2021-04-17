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


def get_doc(sentence: str):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    return doc


def extract_paths_to_token(sentence: str) -> dict:

    doc = get_doc(sentence)

    head_of = dict()
    for token in doc:
        head_of[token] = token.head

    root = find_root(doc)
    head_of[root] = None
    paths = dict()

    for token in doc:
        temp = token
        path = list()

        while temp is not None:
            path.append(temp)
            temp = head_of[temp]

        path.reverse()
        paths[token] = path

    return paths


def find_root(doc):
    for token in doc:
        if token.dep_ == "ROOT":
            return token


def extract_subtrees(sentence: str) -> dict:

    doc = get_doc(sentence)
    trees = dict()

    for token in doc:
        trees[token] = token.subtree

    return trees

