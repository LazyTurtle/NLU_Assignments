import my_module
import spacy
# from spacy import displacy

if __name__ == '__main__':
    print("---start---")
    text = my_module.clean_text("data/text.txt")

    paths = my_module.extract_paths_to_token(text[3])

    for token, path in paths.items():
        print("The path for '{}' is: {}".format(token, path))
