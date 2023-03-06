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

# distribution = {}
#
# with open('creation_of_dataset/abstracts_good.pickle', 'rb') as file:
#     abstracts_good = pickle.load(file)
#     # Loop over the good_abstracts
#     for i, line in enumerate(abstracts_good):
#         # Add the journal to the distribution dictionary to count the number of occurrences
#         distribution[fetch_journal(abstracts_good[line][0])] = distribution.get(fetch_journal(abstracts_good[line][0]), 0) + 1
#         print(i)

dist = {'BMC Genomics': 7, 'Appl Environ Microbiol': 130, 'Mycopathologia': 2, 'Appl Microbiol Biotechnol': 26, 'AMB Express': 3, 'Mycologia': 1, 'Chem Biol': 139, 'J Am Chem Soc': 103, 'Toxicon': 2, 'Acta Crystallogr D Biol Crystallogr': 2, 'Biochemistry': 43, 'J Bacteriol': 126, 'Proc Natl Acad Sci U S A': 74, 'Org Biomol Chem': 8, 'Tetrahedron': 4, 'J Mol Biol': 10, 'Chembiochem': 81, 'Angew Chem Int Ed Engl': 18, 'Microbiology (Reading)': 97, 'J Biol Chem': 61, 'Mol Microbiol': 45, 'Antimicrob Agents Chemother': 47, 'Chem Commun (Camb)': 9, 'Science': 9, 'Biochem Biophys Res Commun': 4, 'Medchemcomm': 5, 'Arch Microbiol': 13, 'Acta Biochim Pol': 1, 'Mol Genet Genomics': 11, 'Fungal Genet Biol': 14, 'J Biotechnol': 5, 'Mol Plant Microbe Interact': 12, 'Glycobiology': 4, 'J Antibiot (Tokyo)': 40, 'J Nat Prod': 20, 'J Ind Microbiol Biotechnol': 10, 'Lett Appl Microbiol': 2, 'Nat Biotechnol': 8, 'Biotechnol Prog': 2, 'Mol Gen Genet': 25, 'Gene': 31, 'Nat Struct Biol': 1, 'FEBS Lett': 8, 'Nature': 12, 'Mol Biosyst': 7, 'J Agric Food Chem': 3, 'Eukaryot Cell': 2, 'mBio': 3, 'FEMS Microbiol Lett': 28, 'J Microbiol Biotechnol': 4, 'ACS Chem Biol': 28, 'Chemistry': 2, 'Org Lett': 29, 'Biosci Biotechnol Biochem': 20, 'J Med Chem': 2, 'Food Technol Biotechnol': 1, 'Biotechnol Lett': 5, 'Sci Rep': 3, 'Nat Chem Biol': 29, 'Phytochemistry': 2, 'ACS Synth Biol': 4, 'Int J Food Microbiol': 3, 'Mol Plant Pathol': 2, 'Arch Toxicol': 1, 'Nat Chem': 2, 'Mar Drugs': 2, 'Mol Cells': 3, 'Folia Microbiol (Praha)': 3, 'Biochim Biophys Acta': 2, 'Curr Opin Chem Biol': 1, 'Plant Cell': 2, 'PLoS One': 13, 'Microb Biotechnol': 4, 'Bioorg Med Chem Lett': 4, 'Acc Chem Res': 1, 'Bioorg Med Chem': 1, 'Nat Struct Mol Biol': 1, 'Cell': 4, 'Carbohydr Res': 1, 'Microb Cell Fact': 1, 'Metab Eng': 2, 'EMBO J': 2, 'Genetika': 1, 'J Appl Microbiol': 3, 'Mol Biol Evol': 2, 'Genome Announc': 6, 'Nat Prod Rep': 1, 'BMC Microbiol': 4, 'PLoS Genet': 1, 'ISME J': 3, 'Eur J Biochem': 11, 'Trends Microbiol': 1, 'J Mol Microbiol Biotechnol': 4, 'Nat Microbiol': 1, 'FEBS J': 1, 'Chem Sci': 3, 'J Org Chem': 2, 'Environ Microbiol': 9, 'Res Microbiol': 4, 'Biotechnology (N Y)': 1, 'Biotechnol Bioeng': 1, 'Wei Sheng Wu Xue Bao': 2, 'Front Microbiol': 1, 'Can J Microbiol': 5, 'Curr Protein Pept Sci': 1, 'J Dairy Sci': 1, 'DNA Seq': 1, 'J Biosci Bioeng': 1, 'Int J Microbiol': 1, 'Infect Immun': 7, 'J Biochem': 2, 'J Gen Microbiol': 1, 'PLoS Biol': 2, 'Anal Biochem': 1, 'Biopolymers': 5, 'J Chem Ecol': 1, 'Biotechnol J': 1, 'Rapid Commun Mass Spectrom': 1, 'Environ Sci Technol': 1, 'Plant J': 3, 'Nat Commun': 2, 'Methods Enzymol': 8, 'Antonie Van Leeuwenhoek': 2, 'Arch Biochem Biophys': 3, 'Curr Microbiol': 2, 'Vet Microbiol': 9, 'J Clin Microbiol': 12, 'Biochem J': 2, 'Int J Med Microbiol': 2, 'Microbiol Immunol': 1, 'Glycoconj J': 2, 'Plant Physiol': 1, 'Genetics': 1, 'Genome Biol Evol': 1, 'Plasmid': 1, 'Genet Mol Res': 2, 'Crit Rev Microbiol': 1, 'Acta Crystallogr Sect F Struct Biol Cryst Commun': 4, 'Biomacromolecules': 1, 'Future Microbiol': 1, 'BMC Biochem': 1, 'Brief Funct Genomics': 1, 'Appl Biochem Biotechnol': 1, 'FEMS Microbiol Ecol': 1, 'PLoS Pathog': 1, 'Beilstein J Org Chem': 1, 'Mol Pharmacol': 1, 'Cell Chem Biol': 1, 'Curr Genet': 1}

# sort the dictionary by value
dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1], reverse=True)}

# remove entries with less than 5 publications
dist = {k: v for k, v in dist.items() if v >= 5}

# plot the distribution
plt.bar(range(len(dist)), list(dist.values()), align='center')
plt.xticks(range(len(dist)), list(dist.keys()), rotation=90, fontsize=6)
plt.xlabel('Journals', fontsize=8)
plt.ylabel('Number of publications', fontsize=8)
plt.title('Distribution of journals', fontsize=10)
plt.tight_layout()
plt.savefig('distribution.png', bbox_inches='tight', dpi=300)
