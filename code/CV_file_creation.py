import random
import numpy as np
import os

def read_in_chunks(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line

def write_to_file(file_name, data):
    with open(file_name, 'w') as f:
        for line in data:
            f.write(line)

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
# dataset_protein_text_v2_shorten
data = read_in_chunks('dataset_protein_peptides_complete.txt')
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
    test_file_name = f'test_dataset_protein_v2_{i}.txt'
    train_file_name = f'train_dataset_protein_v2_{i}.txt'

    write_to_file(test_file_name, test_data)
    write_to_file(train_file_name, train_data)
