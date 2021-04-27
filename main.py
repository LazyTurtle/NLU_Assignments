import assignment_module_1
import assignment_module_2
import conll


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
    print("", "")
    docs, spacy_estimates, conll_dataset = assignment_module_2.extract_data("data/conll2003/test.txt")

    pos_spacy_list = assignment_module_2.build_simple_data_list(spacy_estimates, 1)
    pos_conll_list = assignment_module_2.build_simple_data_list(conll_dataset, 1)

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

    print("", "")
    iob_spacy_list = assignment_module_2.build_simple_data_list(spacy_estimates, 3)
    iob_conll_list = assignment_module_2.build_simple_data_list(conll_dataset, 3)

    iob_acc, iob_tot_acc, iob_tot, iob_accuracies = assignment_module_2.evaluate_lists(iob_spacy_list, iob_conll_list)

    print("--------Named entities--------")
    print("Token level named entities total accuracy: {}".format(iob_acc))
    print("Accurate predictions: {}. Total attempts {}".format(iob_tot_acc, iob_tot))
    for tag in sorted(iob_accuracies.keys()):
        print("Accuracy for tag '{}': {}".format(tag, iob_accuracies[tag]))
    print("", "")

    spacy_list = assignment_module_2.build_grouped_data_list(spacy_estimates, 3)
    conll_list = assignment_module_2.build_grouped_data_list(conll_dataset, 3)

    print("Chunk level named entities evaluation")
    per_chunk_evaluation = conll.evaluate(conll_list, spacy_list)
    for tag in sorted(per_chunk_evaluation.keys()):
        print(tag)
        print("Precision: {}".format(per_chunk_evaluation[tag]["p"]))
        print("Recall: {}".format(per_chunk_evaluation[tag]["r"]))
        print("F-measure: {}".format(per_chunk_evaluation[tag]["f"]))
    print("", "")

    combinations = list()
    frequency = dict()
    for doc in docs:
        entities = assignment_module_2.group_entities(doc)
        if entities is not None:
            for e in entities:
                combinations.append(" ".join(e))
    for combination in set(combinations):
        frequency[combination] = 0
    for combination in combinations:
        frequency[combination] += 1
    sorted_combination = sorted(frequency, key=frequency.get)
    sorted_combination.reverse()

    print("Frequency of named entities types")
    for key in sorted_combination:
        print("Frequency of combination:", key)
        print(frequency[key])
    print("", "")

    extended_noun_compounds = assignment_module_2.extend_noun_compound(docs)
    print(extended_noun_compounds)


if __name__ == '__main__':
    first_assignment()
    second_assignment()
