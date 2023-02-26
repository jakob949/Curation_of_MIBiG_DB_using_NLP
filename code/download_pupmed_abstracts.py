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
        abstracts[pmid] = abstract

        print(i)
        if i == 25:
            break
    with open("../pmid_abstract.json", "w") as outfile:
        json.dump(abstracts, outfile)
    return abstracts, broken_files


abstracts = loop_through_pmid_list()[0]

# files missing PubMed IDs
broken_files_in_MiBig = loop_through_pmid_list()[1]


# open the json file and print the abstract for a given PubMed ID - TESTING
with open("../pmid_abstract.json") as f_in:
    abstracts = json.load(f_in)
    print(abstracts["10449723"])