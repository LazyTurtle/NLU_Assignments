import my_module

if __name__ == '__main__':
    print("---start---")
    text = my_module.clean_text("data/text.txt")

    test_sentence = text[4]

    paths = my_module.extract_paths_to_token(test_sentence)

    for token, path in paths.items():
        print("The path for '{}' is: {}".format(token, path))

    trees = my_module.extract_subtrees(test_sentence)

    for token, tree in trees.items():
        print("The subtree of '{}' is: {}".format(token, list(tree)))

    random_attempts = 5
    for i in range(random_attempts):
        token_list = my_module.get_token_list(test_sentence)
        if my_module.token_to_subtree_check(token_list, test_sentence):
            print("The list {} is a subtree".format(token_list))
        else:
            print("The list {} is NOT a subtree".format(token_list))

    span = my_module.get_token_list(test_sentence)
    root_of_head = my_module.extract_head_of_span(span, test_sentence)
    print("The root of the span '{}' is '{}'".format(span, root_of_head))

    dependency_info = my_module.extract_nsubj_dobj_iobj(test_sentence)
    print(dependency_info)
