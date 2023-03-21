#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter
import spacy
import pickle
from sklearn.metrics import accuracy_score


def majority_voting_classifier(preds):
    """
    Given a list of predictions, return the majority vote and accuracy.
    """
    # Count the number of predictions for each label
    if preds is None:
        with open('dataset_positives_titles_abstracts.txt') as f:
            positives = f.readlines()
        with open('dataset_negatives_titles_abstracts.txt') as f:
            negatives = f.readlines()

        complete_dataset = positives + negatives
        preds = []
        for line in complete_dataset:
            preds.append(line.split('\t')[1][:-1])


    counts = Counter(preds)
    # accuracy is the number of correct predictions divided by the total number of predictions
    accuracy = max(counts.values()) / len(preds)
    # majority vote is the label that appears most often
    majority_vote = counts.most_common(1)[0][0]
    return majority_vote, accuracy

def most_common_words_with_relative_frequencies(document: str, n: int) -> list:

    freqs = {}

    document = document.replace("Title:", "").replace("Abstract:", "")

    # Divide the document into chunks
    chunk_size = 1000000

    doc_chunks = [document[i:i + chunk_size] for i in range(0, len(document), chunk_size)]

    # Count the occurrences of the lemmas
    word_counter = Counter()
    for chunk in doc_chunks:
        # Process the chunk using spaCy
        doc = nlp(chunk)

        for token in doc:
            # Ignore stopwords, punctuation, whitespace, and words shorter than 3 characters
            if not token.is_stop and not token.is_punct and not token.is_space and len(token.lemma_) > 2:
                lemma = token.lemma_
                word_counter[lemma] += 1

    # Get the n most common lemmas and their counts
    most_common_lemmas = word_counter.most_common(n)

    # Calculate the total number of counted words
    total_words = sum(word_counter.values())

    # Calculate the relative frequencies and store them in a list of tuples
    most_common_lemmas_with_relative_frequencies = [(word, count / total_words) for word, count in most_common_lemmas]

    freqs = dict(most_common_lemmas_with_relative_frequencies)

    # Return a list of tuples with the most common words and their relative frequencies
    return most_common_lemmas_with_relative_frequencies, word_counter, freqs
def classifier_based_on_freq_of_words(fold):


    with open(f'freqs_{fold}_pos.pck', 'rb') as handle:
        freqs_pos = pickle.load(handle)

    with open(f'freqs_{fold}_neg.pck', 'rb') as handle:
        freqs_neg = pickle.load(handle)


    with open(f'test_fold_{fold}.txt') as test_fold:
        predicitons = []
        labels = []
        for i, line in enumerate(test_fold):
            pos_counter, neg_counter = 0, 0
            label = line.split('\t')[1][:-1]
            labels.append(label)
            most_common_lemmas_with_relative_frequencies, word_counter, freqs = most_common_words_with_relative_frequencies(line, 1000)
            for word, freq in freqs.items():
                if word in freqs_pos and word in freqs_neg:
                    # print('label', label, "POSITIVE", freqs_pos[word], "NEGATIVE", freqs_neg[word], "DIFFERENCE", freqs_pos[word] - freqs_neg[word], "FREQ", freq)
                    if freqs_pos[word] >= freqs_neg[word]:
                        pos_counter += 1
                    else:
                        neg_counter += 1
                elif word in freqs_pos and word not in freqs_neg:
                    pos_counter += 1
                elif word not in freqs_pos and word in freqs_neg:
                    neg_counter += 1

            if pos_counter >= neg_counter:
                predicitons.append('1')
            else:
                predicitons.append('0')

    print('accuracy', accuracy_score(labels, predicitons))

nlp = spacy.load("en_core_web_sm")
classifier_based_on_freq_of_words('4')
# print(majority_voting_classifier(None))
