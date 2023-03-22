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
from pybliometrics.scopus import AbstractRetrieval

try:
    with open('creation_of_dataset/abstracts_bad.pickle', 'rb') as f:
        abstracts_bad = pickle.load(f)
    with open('creation_of_dataset/truly_broken_files_subset_of_broken_files_doi_or_no_pmid.pickle', 'rb') as f:
        truly_broken_files_subset_of_broken_files_doi_or_no_pmid = pickle.load(f)
    with open('creation_of_dataset/abstracts_good.pickle', 'rb') as f:
        abstracts_good = pickle.load(f)
    with open('creation_of_dataset/abstracts_bad_true_09032023.pickle', 'rb') as f:
        abstracts_bad_true_09032023 = pickle.load(f)
    with open('creation_of_dataset/index.pickle', 'rb') as f:
        index = pickle.load(f)


except:
    abstracts_bad_true_09032023 = {}
    index = 0



def predict(abstract_text, tokenizer, model):
    with torch.no_grad():
        inputs = tokenizer(abstract_text, padding=True, truncation=True, return_tensors="pt")
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        predictions = torch.argmax(outputs.logits, dim=1)
        print('Predicted class:', predictions.item(), '\tprobs', torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0])
        return predictions.item()



def loop_through_pmid_list(start_index, abstracts_good = {}, abstracts_bad = {}, truly_broken_files_subset_of_broken_files_doi_or_no_pmid = []):
    """
       Loop through the files in /mibig-json/data/
       retrieves the PubMed IDs and fetch the to the function fetch_abstract.
       to get the abstracts. Saves it in a dictionary/json file.
    """


    tokenizer = RobertaTokenizer.from_pretrained('finetuned_model_roberta_4')
    model = RobertaForSequenceClassification.from_pretrained('finetuned_model_roberta_4')


    for j, item in enumerate(abstracts_bad):
        if j >= start_index:
            abstract = abstracts_bad[item][1]
            ID = abstracts_bad[item][0]

            # Store the abstract in the dictionary
            predict(abstract.replace('\n', ' ').replace('\t', ' '), tokenizer, model)
            abstract = abstract.replace('\n', ' ').replace('\t', ' ')

            try:
                ab = AbstractRetrieval(ID)
                title = ab.title
            except:
                print('something went wrong with the title or ID')
                truly_broken_files_subset_of_broken_files_doi_or_no_pmid.append(item)
                continue


            classification = input(f"i={j}, TITLE: {title}... ABSTRACT: {abstract}")
            if classification == '1':

                abstracts_good[f"{item}"] = [ID, abstract]
            elif classification == '0':

                abstracts_bad_true_09032023[f"{item}"] = [ID, abstract]
            elif classification != '1' or classification != '0' or classification == 'stop':
                print('Wrong input or stopped')
                break


    return abstracts_good, abstracts_bad_true_09032023, j, truly_broken_files_subset_of_broken_files_doi_or_no_pmid

print(index)
abstracts_good, abstracts_bad_true_09032023, index, truly_broken_files_subset_of_broken_files_doi_or_no_pmid = loop_through_pmid_list(start_index = index, abstracts_good=abstracts_good, abstracts_bad=abstracts_bad, truly_broken_files_subset_of_broken_files_doi_or_no_pmid=truly_broken_files_subset_of_broken_files_doi_or_no_pmid)


# Saves results as pickel files
with open('creation_of_dataset/abstracts_good.pickle', 'wb') as f:
    pickle.dump(abstracts_good, f)
with open('creation_of_dataset/abstracts_bad_true_09032023.pickle', 'wb') as f:
    pickle.dump(abstracts_bad_true_09032023, f)
with open('creation_of_dataset/truly_broken_files_subset_of_broken_files_doi_or_no_pmid.pickle', 'wb') as f:
    pickle.dump(truly_broken_files_subset_of_broken_files_doi_or_no_pmid, f)
with open('creation_of_dataset/index.pickle', 'wb') as f:
    pickle.dump(index, f)