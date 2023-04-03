import random
import numpy as np
# Read in the data from the two labeled files
with open('dataset_negatives_titles_abstracts.txt', 'r') as f:
    data_0 = f.readlines()
with open('dataset_positives_titles_abstracts.txt', 'r') as f:
    data_1 = f.readlines()

data = data_0 + data_1

with open ('dataset_protien_chem.txt', 'r') as f:
    data = f.readlines()

# Set the number of folds
k = 5

# shuffle the data
random.shuffle(data)

# Split the data into k folds
folds = np.array_split(data, k)

# Loop over each fold and create test and train sets
for i in range(k):
    # Get the test data for this fold
    test_data = folds[i]

    # Get the training data for this fold
    train_folds = [f for j, f in enumerate(folds) if j != i]
    train_data = np.concatenate(train_folds)

    # Save the test and train data to text files
    with open(f'test_fold_protein_{i}.txt', 'w') as f:
        f.writelines(test_data)
    with open(f'train_fold_protein_{i}.txt', 'w') as f:
        f.writelines(train_data)
