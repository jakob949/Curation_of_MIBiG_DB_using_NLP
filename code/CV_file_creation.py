import random
import numpy as np
import os

def read_in_chunks(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line

def combine_data(file_paths):
    for file_path in file_paths:
        for line in read_in_chunks(file_path):
            yield line

def split_into_folds(data, k):
    folds = [list() for _ in range(k)]
    i = 0
    for line in data:
        folds[i].append(line)
        i = (i + 1) % k
    return folds

def write_to_file(file_name, data):
    with open(file_name, 'w') as f:
        for line in data:
            f.write(line)

def split_into_folds(data, k):
    folds = [list() for _ in range(k)]
    i = 0
    for line in data:
        folds[i].append(line)
        i = (i + 1) % k
    return folds

#
# data = read_in_chunks('dataset/protein_SMILE/dataset_protein_peptides_complete_v3.txt')
#
# random.seed(1)
#
# shuffled_data = []
# for line in data:
#     shuffled_data.insert(random.randrange(len(shuffled_data) + 1), line)
#
# k = 6
#
# folds = split_into_folds(shuffled_data, k)
#
# for i in range(k):
#     test_data = folds[i]
#
#     remaining_folds = [f for j, f in enumerate(folds) if j != i]
#
#     # Split remaining folds into train and validation
#     validation_data = remaining_folds.pop(i % len(remaining_folds))
#
#     train_data = []
#     for fold in remaining_folds:
#         train_data.extend(fold)
#
#     train_file_name = f'dataset/protein_SMILE/train_protein_peptides_complete_v3_4_shorten_{i}.txt'
#     validation_file_name = f'dataset/protein_SMILE/validation_protein_peptides_complete_v3_4_shorten_{i}.txt'
#     test_file_name = f'dataset/protein_SMILE/test_protein_peptides_complete_v3_4_shorten_{i}.txt'
#
#     write_to_file(test_file_name, test_data)
#     write_to_file(train_file_name, train_data)
#     write_to_file(validation_file_name, validation_data)
#     break

# dataset_protein_text_v2_shorten
data = read_in_chunks('dataset/geneProduct2SMILE/dataset_geneProduct2SMILES_v1.txt')
# Read in the data from the two labeled files
random.seed(1)
# Shuffle the data
shuffled_data = []
for line in data:
    shuffled_data.insert(random.randrange(len(shuffled_data) + 1), line)

# Set the number of folds
k = 5

# Split the data into k folds
folds = split_into_folds(shuffled_data, k)

# Loop over each fold and create test and train sets
for i in range(k):
    # Get the test data for this fold
    test_data = folds[i]

    # Get the training data for this fold
    train_folds = [f for j, f in enumerate(folds) if j != i]
    train_data = []
    for fold in train_folds:
        train_data.extend(fold)

    # Save the test and train data to text files
    test_file_name = f'dataset/geneProduct2SMILE/test_geneproduct2SMILES_{i}.txt'
    train_file_name = f'dataset/geneProduct2SMILE/train_geneproduct2SMILES_{i}.txt'

    write_to_file(test_file_name, test_data)
    write_to_file(train_file_name, train_data)
    break



#### CV for multitask dataset

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', names=['input_text', 'label'])
    return data


def split_data(data, test_size=0.2):
    train_data = pd.DataFrame(columns=['input_text', 'label'])
    test_data = pd.DataFrame(columns=['input_text', 'label'])

    # Group data by labels
    grouped = data.groupby('label')

    # For each group, assign it completely to either train_data or test_data
    for name, group in grouped:
        if np.random.rand() < test_size:  # This label goes into the test set
            test_data = pd.concat([test_data, group])
        else:  # This label goes into the training set
            train_data = pd.concat([train_data, group])

    return train_data, test_data  # Corrected line



def multi():
    # Load data
    data = load_data("dataset/Multitask/dataset_multitask_v2.txt")

    # Split data
    train_data, test_data = split_data(data)

    # Save train and test data
    train_data.to_csv("train_data_mult.txt", sep='\t', index=False)
    test_data.to_csv("test_data_mult.txt", sep='\t', index=False)

#multi()
