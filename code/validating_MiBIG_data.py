import os
import re
import json
import pickle
import requests
from bs4 import BeautifulSoup
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModel, AutoTokenizer,
                          AutoModelForMaskedLM, RobertaForSequenceClassification,
                          RobertaTokenizer, BertTokenizer, BertForSequenceClassification, AdamW)


def fetch_abstract(pubmed_id):
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    abstract_section = soup.find("div", {"class": "abstract-content selected"})
    if abstract_section is None:
        return None
    abstract = abstract_section.text.strip()
    return abstract


def predict(abstract_text, tokenizer, model):
    with torch.no_grad():
        inputs = tokenizer(abstract_text, padding=True, truncation=True, return_tensors="pt")
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        predictions = torch.argmax(outputs.logits, dim=1)
        print('Prediction class:', predictions.item(), '\tprobs', torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0])
        return predictions.item()


def loop_through_pmid_list(start_index = 0, abstracts_good = {}, abstracts_bad = {}, broken_files = []):
    """
       Loop through the files in /mibig-json/data/
       retrieves the PubMed IDs and fetch the to the function fetch_abstract.
       to get the abstracts. Saves it in a dictionary/json file.
    """
    classification = ''
    abstact_list = []
    tokenizer = RobertaTokenizer.from_pretrained('finetuned_roberta_epoch8')
    model = RobertaForSequenceClassification.from_pretrained('finetuned_roberta_epoch8')
    for i, filename in enumerate(os.listdir("../mibig-json/data/")):
        if filename.endswith(".json") and i >= start_index:
            with open(f"../mibig-json/data/{filename}") as f_in:
                file = json.load(f_in)
                try:
                    pubmed_list = file["cluster"]["publications"]
                    print(pubmed_list, '\ti = ', i)
                except:
                    broken_files.append(filename)
                    continue


            for j, item in enumerate(pubmed_list):
                # Fetch the abstract for the PubMed ID
                abstract = fetch_abstract(item.replace("pubmed:", ""))
                # Store the abstract in the dictionary
                if abstract is not None:
                    predict(abstract.replace('\n', ' ').replace('\t', ' '), tokenizer, model)
                    classification = input(abstract.replace('\n', ' ').replace('\t', ' '))
                    if classification == '1':

                        abstracts_good[f"{filename}_{j}"] = [item.replace("pubmed:", ""), abstract]
                        pattern = re.compile(r'\s+$')
                        abstact_list.append([re.sub(pattern, ' ', abstract.replace('\n', '')), 1])
                    elif classification == '0':
                        abstracts_bad[f"{filename}_{j}"] = [item.replace("pubmed:", ""), abstract]
                        pattern = re.compile(r'\s+$')
                        abstact_list.append([re.sub(pattern, ' ', abstract.replace('\n', '')), 0])
                    elif classification == 'stop':
                        break
                elif abstract is None:
                    broken_files.append(f"{filename}_{j}")
            if classification == 'stop':
                break

    # with open("../pmid_abstract.json", "w") as outfile:
    #     json.dump(abstracts, outfile)
    return abstracts_good, abstracts_bad, broken_files, abstact_list, i


# Opens the pickle files, where the earlier results are stored
try:
    with open('creation_of_dataset/abstracts_good.pickle', 'rb') as f:
        abstracts_good = pickle.load(f)
    with open('creation_of_dataset/abstracts_bad.pickle', 'rb') as f:
        abstracts_bad = pickle.load(f)
    with open('creation_of_dataset/broken_files.pickle', 'rb') as f:
        broken_files = pickle.load(f)
    with open('creation_of_dataset/index.pickle', 'rb') as f:
        index = pickle.load(f)
except:
    abstracts_good = {}
    abstracts_bad = {}
    broken_files = []
    index = 0

abstracts_good, abstracts_bad, broken_files, abstact_list, index = loop_through_pmid_list(start_index = index, abstracts_good=abstracts_good, abstracts_bad=abstracts_bad, broken_files=broken_files)


# Saves results as pickel files
with open('creation_of_dataset/abstracts_good.pickle', 'wb') as f:
    pickle.dump(abstracts_good, f)
with open('creation_of_dataset/abstracts_bad.pickle', 'wb') as f:
    pickle.dump(abstracts_bad, f)
with open('creation_of_dataset/broken_files.pickle', 'wb') as f:
    pickle.dump(broken_files, f)
with open('creation_of_dataset/index.pickle', 'wb') as f:
    pickle.dump(index, f)