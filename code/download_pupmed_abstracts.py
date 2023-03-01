import requests
from bs4 import BeautifulSoup
import os
import json

def fetch_abstract(pubmed_id):
    """
    Fetch the abstract for a given PubMed ID from ncbi.
    """

    # Construct the PubMed URL
    url = "https://pubmed.ncbi.nlm.nih.gov/{}/".format(pubmed_id)

    # Make a GET request to the URL
    response = requests.get(url)
    # Parse the HTML content of the response using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the abstract section and extract the text
    abstract_section = soup.find("div", {"class": "abstract-content selected"})
    if abstract_section is None:
        return None
    #  strips the unnecessary HTML formats from the abstract
    abstract = abstract_section.text.strip()
    return abstract


def loop_through_pmid_list():
    """
       Loop through the files in /mibig-json/data/
       retrieves the PubMed IDs and fetch the to the function fetch_abstract.
       to get the abstracts. Saves it in a dictionary/json file.
    """

    # Create a dictionary to store the abstracts
    abstracts = {}
    broken_files = []
    abstact_list = []
    for i, filename in enumerate(os.listdir("../mibig-json/data/")):
        if filename.endswith(".json"):
            with open(f"../mibig-json/data/{filename}") as f_in:
                file = json.load(f_in)
                try:
                    pmid = file["cluster"]["publications"][0].replace("pubmed:", "")
                except:
                    broken_files.append(filename)
                    print(filename, '\n', file)
                    continue

        # Fetch the abstract for the PubMed ID
        abstract = fetch_abstract(pmid)
        # Store the abstract in the dictionary
        if abstract is not None:
            abstracts[pmid] = abstract
            import re
            pattern = re.compile(r'\s+$')
            abstact_list.append([re.sub(pattern, ' ', abstract.replace('\n', '')), 1])

        print(i)
        if i == 100:
            break

    with open("../pmid_abstract.json", "w") as outfile:
        json.dump(abstracts, outfile)
    return abstracts, broken_files, abstact_list
def loop_pmid_list(file):
    abstact_list = []
    with open(file) as file:
        for i, line in enumerate(file):
            abstact = fetch_abstract(line[:-1])
            if abstact is not None:
                import re
                pattern = re.compile(r'\s+$')
                format_abs = re.sub(pattern, ' ', abstact.replace('\n', ''))
                abstact_list.append([format_abs, 0])
            print(i, format_abs)
        return abstact_list


abstracts_list = loop_pmid_list("testing_small_neg_pubmed.txt")

with open('Testing_small_neg_abstacts.tsv', 'w') as tsvfile:
    for line in abstracts_list:
        tsvfile.write(line[0] + '\t' + str(line[1]) + '\n')

# # open the json file and print the abstract for a given PubMed ID - TESTING
# with open("../pmid_abstract.json") as f_in:
#     abstracts = json.load(f_in)
#     print(abstracts["10449723"])


