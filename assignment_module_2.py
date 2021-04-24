import spacy
from spacy.tokens import Doc
import conll


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


def build_sentence_string(sentence_data_touples:list)->str:
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
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    docs = list()

    for doc in nlp.pipe(sentences):
        docs.append(doc)




    evaluation = 0

    return evaluation
