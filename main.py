import my_module
import spacy
# from spacy import displacy

if __name__ == '__main__':
    print("---start---")
    text = my_module.clean_text("data/text.txt")

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text[2])
    path = my_module.extract_path_to_token(doc, doc[-2])
    print("The path from {} to {} is: {}".format(my_module.find_root(doc), doc[-2], path))
