#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The idea it to know the real distribution of journals used to create the positive dataset
# Based on the results, is it possible create a negative dataset with the same journal distribution
# Hopefully, this will improve the model generalization, and decrease the bias

import pickle
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
def fetch_journal(pubmed_id):
    """
    Fetch the journal for a given PubMed ID from ncbi.
    :param: pubmed_id
    :return: journal_title
    """
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    journal_section = soup.find_all("div", {"class": "journal-actions dropdown-block"})
    journal_title = journal_section[0].text.strip().split('\n')[0]
    return journal_title

distribution = {}

with open('creation_of_dataset/abstracts_good.pickle', 'rb') as file:
    abstracts_good = pickle.load(file)
    # Loop over the good_abstracts
    for i, line in enumerate(abstracts_good):
        # Add the journal to the distribution dictionary to count the number of occurrences
        distribution[fetch_journal(abstracts_good[line][0])] = distribution.get(fetch_journal(abstracts_good[line][0]), 0) + 1


# Plot the distribution of journals
plt.bar(distribution.keys(), distribution.values())
plt.xticks(rotation=45)
plt.show()
