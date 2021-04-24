import spacy
from spacy.tokens import Doc
import conll


class ConfusionMatrix:
    def __init__(self):
        true_positive = int()
        true_negative = int()
        false_positive = int()
        false_negative = int()


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)


def build_word_data_list(corpus_line: tuple):
    word_list = list()
    for word_data_tuple in corpus_line:
        for data in word_data_tuple:
            word, pos_tag, chunk_tag, named_entity = data.split()
            word_list.append((word, pos_tag, chunk_tag, named_entity))
    return word_list


def build_sentence_string(sentence_data_touples: list) -> str:
    words = [word for word, pos_tag, chunk_tag, named_entity in sentence_data_touples]
    return " ".join(words)


def build_space_list(word_list: list) -> list:
    spaces = [False] * len(word_list)
    punct_no_space_before = [".", "!", "?", ",", ";", ":", "'", ")"]
    punct_no_space_after = ["("]

    for i in range(len(word_list) - 1):
        if word_list[i + 1][0] in punct_no_space_before or word_list[i] in punct_no_space_after:
            spaces[i] = False
        else:
            spaces[i] = True

    return spaces


def corrected_spacy_tag(converter: dict, spacy_tag: str) -> str:
    if converter is None:
        return spacy_tag
    if spacy_tag in converter.keys():
        return converter[spacy_tag]
    else:
        return spacy_tag


def calculate_accuracy(tag: str, docs: list, dataset: list, spacy_tag_converter=None):
    accurate_predictions = 0
    total_prediction = 0
    assert len(docs) == len(dataset), \
        "The number of sentences should be equal. Docs({}) dataset({})".format(len(docs), len(dataset))

    for i in range(len(docs)):
        assert len(docs[i]) == len(dataset[i]), \
            "The number of tokens should be equal. Docs{}({}) dataset{}({})".format(i, len(docs), i, len(dataset))

        for j in range(len(docs[i])):
            true_tag = dataset[i][j][1]
            if true_tag != tag:
                continue

            spacy_tag = corrected_spacy_tag(spacy_tag_converter, docs[i][j].tag_)
            total_prediction += 1

            accurate_predictions += 1 if true_tag == spacy_tag else 0
            if true_tag == tag and spacy_tag != true_tag:
                print(docs[i][j], spacy_tag, true_tag)
    accuracy = accurate_predictions / total_prediction if total_prediction > 0 else None

    return accuracy, accurate_predictions, total_prediction


def evaluate_spacy_ner(data):
    dataset = list()
    sentences = list()
    corpus = conll.read_corpus_conll(data)

    for sent in corpus:
        dataset.append(build_word_data_list(sent))

    for sent_tuples in dataset:
        if sent_tuples[0][0] == '-DOCSTART-':
            dataset.remove(sent_tuples)

    for sent_tuples in dataset:
        sentences.append(build_sentence_string(sent_tuples))

    nlp = spacy.load("en_core_web_sm")
    # I need this WhitespaceTokenizer otherwise spacy and conll are not in sync
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    pos_label = [label for label in nlp.get_pipe("tagger").labels]
    # to convert during the accuracy calculation
    pos_label.append("(")
    pos_label.append(")")
    pos_label.append('"')
    pos_label.append("''")

    docs = list()
    for doc in nlp.pipe(sentences):
        docs.append(doc)

    convert_tag = dict()
    convert_tag["-LRB-"] = "("
    convert_tag["-RRB-"] = ")"
    convert_tag["HYPH"] = ":"
    convert_tag["``"] = '"'
    convert_tag["''"] = '"'
    accuracies = dict()
    for tag in pos_label:
        accuracies[tag] = calculate_accuracy(tag, docs, dataset, convert_tag)

    return accuracies
