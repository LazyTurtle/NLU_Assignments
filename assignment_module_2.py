import spacy
from spacy.tokens import Doc
import conll


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)


def extract_data(file_path):
    dataset = list()
    sentences = list()
    corpus = conll.read_corpus_conll(file_path)

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

    docs = list()
    for doc in nlp.pipe(sentences):
        docs.append(doc)

    ent_tag_converter = dict()
    ent_tag_converter["PERSON"] = "PER"
    ent_tag_converter["ORG"] = "ORG"
    ent_tag_converter["NORP"] = "ORG"
    ent_tag_converter["FAC"] = "LOC"
    ent_tag_converter["GPE"] = "LOC"
    ent_tag_converter["EVENT"] = "MISC"
    ent_tag_converter["WORK_OF_ART"] = "MISC"
    ent_tag_converter["LANGUAGE"] = "MISC"

    spacy_data = list()
    for i in range(len(dataset)):
        sentence = list()
        for j in range(dataset[i]):
            text = docs[i][j].text
            pos = docs[i][j].pos_
            chunk = ""  # TO DO

            if docs[i][j].ent_type_ in ent_tag_converter.keys():
                iob_tag = docs[i][j].ent_iob_ + "-" + ent_tag_converter[docs[i][j].ent_type_]
            else:
                iob_tag = "O"
            sentence.append((text, pos, chunk, iob_tag))
        spacy_data.append(sentence)

    return


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


def calculate_accuracy_pos(tag: str, docs: list, dataset: list, spacy_tag_converter=None):
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

    accuracy = accurate_predictions / total_prediction if total_prediction > 0 else None

    return accuracy, accurate_predictions, total_prediction


def evaluate_lists(estimates: list, ground_truths: list) -> (float, int, int, dict):
    accurate_predictions = 0
    total_prediction = 0
    per_tag_accuracies = dict()
    tags = set([tag for text, tag in ground_truths])

    for tag in tags:
        tag_acc_pred = 0
        tag_tot_pred = 0
        for i in range(len(estimates)):
            text, tru_tag = ground_truths[i]
            if tru_tag != tag:
                continue
            tag_tot_pred += 1
            tag_acc_pred += 1 if estimates[i][1] == ground_truths[i][1] else 0

        accurate_predictions += tag_acc_pred
        total_prediction += tag_tot_pred
        per_tag_accuracies[tag] = tag_acc_pred / tag_tot_pred if tag_tot_pred > 0 else 0

    accuracy = accurate_predictions / total_prediction if total_prediction > 0 else 0
    return accuracy, accurate_predictions, total_prediction, per_tag_accuracies


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

    pos_tag_converter = dict()
    pos_tag_converter["-LRB-"] = "("
    pos_tag_converter["-RRB-"] = ")"
    pos_tag_converter["HYPH"] = ":"
    pos_tag_converter["``"] = '"'
    pos_tag_converter["''"] = '"'
    accuracies = dict()
    for tag in pos_label:
        accuracies[tag] = calculate_accuracy_pos(tag, docs, dataset, pos_tag_converter)

    ent_tag_converter = dict()
    ent_tag_converter["PERSON"] = "PER"
    ent_tag_converter["ORG"] = "ORG"
    ent_tag_converter["NORP"] = "ORG"
    ent_tag_converter["FAC"] = "LOC"
    ent_tag_converter["GPE"] = "LOC"
    ent_tag_converter["EVENT"] = "MISC"
    ent_tag_converter["WORK_OF_ART"] = "MISC"
    ent_tag_converter["LANGUAGE"] = "MISC"

    hyp = list()
    ref = list()
    for i in range(len(docs)):
        for j in range(len(docs[i])):
            if docs[i][j].ent_type_ in ent_tag_converter.keys():
                hyp_tag = docs[i][j].ent_iob_ + "-" + ent_tag_converter[docs[i][j].ent_type_]
            else:
                hyp_tag = "O"

            hyp.append((docs[i][j].text, hyp_tag))
            ref.append((dataset[i][j][0], dataset[i][j][3]))
    a, b, c, d = evaluate_lists(hyp, ref)
    print(a, d)

    return accuracies
