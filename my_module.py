import spacy
import random


def check_str(string: str):
    assert string.strip() != "", "string must not be an empty string"


# return a list of lists containing the tokens.
def clean_text(file_path: str) -> list:
    check_str(file_path)

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
    check_str(sentence)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    return doc


def extract_paths_to_tokens(sentence: str) -> dict:
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
    assert doc is not None, "doc must not be None"
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
            temp = get_list_from_tree(token.subtree)
            temp.remove(token)
            trees[token] = temp

    return trees


def token_to_subtree_check(token_list: list, sentence: str) -> bool:
    trees = extract_subtrees(sentence)
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
    if -1 < start < end:
        length = end - start
    else:
        # for testing purposes we may want a random contiguous sequence of tokens
        length = random.randint(1, len(doc))
        start = random.randint(0, len(doc) - length)

    token_list = list()
    for i in range(start, start + length):
        token_list.append(doc[i])
    return token_list


def contains_list(list_container: list, list_contained: list):
    print(list_container, list_contained)
    len_container = len(list_container)
    len_contained = len(list_contained)
    assert len_container >= len_contained, "The list container must be equal or larger than the one contained"
    for i in range(len_container - len_contained + 1):
        j = len_contained + i
        if list_container[i:j] == list_contained:
            return i
    return None


def extract_head_of_span(input_span, sentence):
    doc = get_doc(sentence)
    index = contains_list([token.text for token in doc], [token.text for token in input_span])
    if index:
        span = doc[index:len(input_span) + index]
        return span.root
    return None


def extract_nsubj_dobj_iobj(sentence: str) -> dict:
    return extract_info(sentence, "nsubj", "dobj", "iobj")


def extract_info(sentence, *info) -> dict:
    doc = get_doc(sentence)
    assert len(list(info)) > 0, "There must be dependency information to extract"

    dependency_spans = dict()
    for dependency in list(info):
        dependency_spans[dependency] = list()

    for token in doc:
        if token.dep_ in list(info):
            dependency_spans[token.dep_].append(get_list_from_tree(token.subtree))
    return dependency_spans
