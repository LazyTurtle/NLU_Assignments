import my_module
# import spacy
# from spacy import displacy

if __name__ == '__main__':
    print("---start---")
    text = my_module.clean_text("data/text.txt")

    test_sentence = text[3]

    paths = my_module.extract_paths_to_token(test_sentence)

    for token, path in paths.items():
        print("The path for '{}' is: {}".format(token, path))

    trees = my_module.extract_subtrees(test_sentence)

    for token, tree in trees.items():
        print("The subtree of '{}' is: {}".format(token, list(tree)))

    found_sub_tree = False
    while not found_sub_tree:
        token_list = my_module.get_token_list(test_sentence)
        if my_module.token_to_subtree_check(token_list, test_sentence):
            print("The list {} is a subtree".format(token_list))
            found_sub_tree = True
        else:
            print("The list {} is NOT a subtree".format(token_list))
