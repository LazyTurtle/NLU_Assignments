import spacy
from spacy.tokens import Doc
import conll


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)


def extract_data(file_path: str) -> (list, list, list):
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
        for j in range(len(dataset[i])):
            text = docs[i][j].text
            pos = docs[i][j].tag_
            chunk = ""  # TO DO

            if docs[i][j].ent_type_ in ent_tag_converter.keys():
                iob_tag = docs[i][j].ent_iob_ + "-" + ent_tag_converter[docs[i][j].ent_type_]
            else:
                iob_tag = "O"
            sentence.append((text, pos, chunk, iob_tag))
        spacy_data.append(sentence)

    return docs, spacy_data, dataset


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


def evaluate_lists(estimates: list, ground_truths: list, *, spacy_to_conll=None, conll_to_spacy=None) -> (float, int, int, dict):
    assert len(estimates) == len(ground_truths), \
        "The number of items should be equal. ({}) ({})".format(len(estimates), len(ground_truths))
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
            est_tag = estimates[i][1]

            if spacy_to_conll is not None:
                est_tag = spacy_to_conll[est_tag] if est_tag in spacy_to_conll.keys() else est_tag
            if conll_to_spacy is not None:
                tru_tag = conll_to_spacy[tru_tag] if tru_tag in conll_to_spacy.keys() else tru_tag

            tag_acc_pred += 1 if est_tag == tru_tag else 0

        accurate_predictions += tag_acc_pred
        total_prediction += tag_tot_pred
        per_tag_accuracies[tag] = tag_acc_pred / tag_tot_pred if tag_tot_pred > 0 else 0

    accuracy = accurate_predictions / total_prediction if total_prediction > 0 else 0
    return accuracy, accurate_predictions, total_prediction, per_tag_accuracies
