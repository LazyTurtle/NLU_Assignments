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
    print("************** Start second assignment **************")
    docs, spacy_estimates, conll_dataset = assignment_module_2.extract_data("data/conll2003/test.txt")

    pos_spacy_list = list()
    pos_conll_list = list()
    for i in range(len(spacy_estimates)):
        for j in range(len(spacy_estimates[i])):
            spacy_text = spacy_estimates[i][j][0]
            spacy_pos = spacy_estimates[i][j][1]
            conll_text = conll_dataset[i][j][0]
            conll_pos = conll_dataset[i][j][1]
            pos_spacy_list.append((spacy_text, spacy_pos))
            pos_conll_list.append((conll_text, conll_pos))

    pos_tag_converter = dict()
    pos_tag_converter["-LRB-"] = "("
    pos_tag_converter["-RRB-"] = ")"
    pos_tag_converter["HYPH"] = ":"
    pos_tag_converter["``"] = '"'
    pos_tag_converter["''"] = '"'

    pos_acc, pos_tot_acc, pos_tot, pos_accuracies = assignment_module_2.evaluate_lists(pos_spacy_list, pos_conll_list, spacy_to_conll=pos_tag_converter)
    print("--------Part of Speech--------")
    print("Part of speech total accuracy: {}".format(pos_acc))
    print("Accurate predictions: {}. Total attempts {}".format(pos_tot_acc, pos_tot))
    for tag in sorted(pos_accuracies.keys()):
        print("Accuracy for tag '{}': {}".format(tag,pos_accuracies[tag]))

    iob_spacy_list = list()
    iob_conll_list = list()
    for i in range(len(spacy_estimates)):
        for j in range(len(spacy_estimates[i])):
            spacy_text = spacy_estimates[i][j][0]
            spacy_iob = spacy_estimates[i][j][3]
            conll_text = conll_dataset[i][j][0]
            conll_iob = conll_dataset[i][j][3]
            iob_spacy_list.append((spacy_text, spacy_iob))
            iob_conll_list.append((conll_text, conll_iob))

    iob_acc, iob_tot_acc, iob_tot, iob_accuracies = assignment_module_2.evaluate_lists(iob_spacy_list, iob_conll_list)

    print("--------Named entities--------")
    print("Token level named entities total accuracy: {}".format(iob_acc))
    print("Accurate predictions: {}. Total attempts {}".format(iob_tot_acc, iob_tot))
    for tag in sorted(iob_accuracies.keys()):
        print("Accuracy for tag '{}': {}".format(tag, iob_accuracies[tag]))

if __name__ == '__main__':
    # first_assignment()
    second_assignment()
