import spacy
import random


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


def extract_subtrees(sentence: str, include_token=False) -> dict:
    doc = get_doc(sentence)
    trees = dict()

    for token in doc:
        if include_token:
            trees[token] = token.subtree
        else:
            set_tree = set(token.subtree)
            set_token = set()
            set_token.add(token)
            trees[token] = set_tree - set_token

    return trees


def token_to_subtree_check(token_list: list, sentence: str) -> bool:
    trees = extract_subtrees(sentence, True)
    for token, tree in trees.items():
        list_of_tree = get_list_from_tree(tree)
        if [t.text for t in token_list] == [t.text for t in list_of_tree]:
            return True

    return False


def get_list_from_tree(tree):
    token_list = list()
    for token in tree:
        token_list.append(token)
    return token_list


def get_token_list(sentence: str, start=-1, end=-1) -> list:
    doc = get_doc(sentence)
    length = 0
    if -1 < start < end:
        length = end - start
    else:
        # for testing purposes
        length = random.randint(1, len(doc))
        start = random.randint(0, len(doc) - length)

    token_list = list()
    for i in range(start, start + length):
        token_list.append(doc[i])
    return token_list
