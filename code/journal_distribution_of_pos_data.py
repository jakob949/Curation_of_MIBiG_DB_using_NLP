#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The idea it to know the real distribution of journals used to create the positive dataset
# Based on the results, is it possible create a negative dataset with the same journal distribution
# Hopefully, this will improve the model generalization, and decrease the bias
import pickle
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from pybliometrics.scopus import AbstractRetrieval


def fetch_journal(pubmed_id):

    """
    Fetch the journal for a given PubMed ID from ncbi. Very SLOW but works on all journals
    :param: pubmed_id
    :return: journal_title
    """
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    journal_section = soup.find_all("div", {"class": "journal-actions dropdown-block"})
    journal_title = journal_section[0].text.strip().split('\n')[0]
    return journal_title

def fetch_journal2(pubmed_id):
    # faster way to get the journal title
    try:
        ab = AbstractRetrieval(pubmed_id)
        title = ab.publicationName
    except:
        title = fetch_journal(pubmed_id)
        print(title)

        return title
    print(title)
    return title

# dist = {}
# with open('creation_of_dataset/abstracts_good_unique_entries.pickle', 'rb') as file:
#     abstracts_good = pickle.load(file)
#     # Loop over the good_abstracts
#     for i, line in enumerate(abstracts_good):
#         # Add the journal to the distribution dictionary to count the number of occurrences
#         journal = fetch_journal2(abstracts_good[line][0])
#         if journal is not None:
#             dist[journal] = dist.get(journal, 0) + 1
#             # dist[fetch_journal2(abstracts_good[line][0])] = dist.get(fetch_journal2(abstracts_good[line][0]), 0) + 1
#
# # Save the distribution list
# with open('creation_of_dataset/journal_distribution.pickle', 'wb') as file:
#     pickle.dump(dist, file)


with open('creation_of_dataset/journal_distribution.pickle', 'rb') as file:
    dist = pickle.load(file)


# # sort the dictionary by value
dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1], reverse=True)}

for line in dist:
    print(line, dist[line])
    input('Press enter to continue')


# # remove entries with less than 5 publications
# dist = {k: v for k, v in dist.items() if v >= 5}
#
# # plot the distribution
# plt.bar(range(len(dist)), list(dist.values()), align='center')
# plt.xticks(range(0, len(dist), 2), list(dist.keys())[0::2], rotation=90, fontsize=4)
# plt.xlabel('Journals', fontsize=10)
# plt.ylabel('Number of publications', fontsize=10)
# plt.title('Distribution of journals', fontsize=12)
# plt.tight_layout()
# plt.savefig('distribution.png', bbox_inches='tight', dpi=300)


