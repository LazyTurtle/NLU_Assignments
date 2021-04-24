import assignment_module_1
import assignment_module_2


def first_assignment():
    print("---start first assignment---")
    text = assignment_module_1.clean_text("data/text.txt")

    test_sentence = text[4]

    paths = assignment_module_1.extract_paths_to_tokens(test_sentence)

    for token, path in paths.items():
        print("The path for '{}' is: {}".format(token, path))

    trees = assignment_module_1.extract_subtrees(test_sentence)

    for token, tree in trees.items():
        print("The subtree of '{}' is: {}".format(token, list(tree)))

    random_attempts = 5
    for i in range(random_attempts):
        token_list = assignment_module_1.get_token_list(test_sentence)
        if assignment_module_1.token_to_subtree_check(token_list, test_sentence):
            print("The list {} is a subtree".format(token_list))
        else:
            print("The list {} is NOT a subtree".format(token_list))

    span = assignment_module_1.get_token_list(test_sentence)
    root_of_head = assignment_module_1.extract_head_of_span(span, test_sentence)
    print("The root of the span '{}' is '{}'".format(span, root_of_head))

    dependency_info = assignment_module_1.extract_nsubj_dobj_iobj(test_sentence)
    print(dependency_info)


def second_assignment():
    print("---start second assignment---")


if __name__ == '__main__':
    #first_assignment()
    accuracies = assignment_module_2.evaluate_spacy_ner("data/conll2003/test.txt")
    accurate_predictions = 0
    total_predictions = 0
    print(accuracies)
    for key, (fraction, acc, tot) in accuracies.items():
        if fraction is not None:
            accurate_predictions += acc
            total_predictions += tot

    print(accurate_predictions / total_predictions, accurate_predictions, total_predictions)
