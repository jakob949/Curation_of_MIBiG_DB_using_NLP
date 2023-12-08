from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rdkit import Chem
from torchmetrics.text import BLEUScore, ROUGEScore


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return smiles
from torchmetrics.text import ROUGEScore
def rouge_score(pred, true):
    rouge.update(pred, true)
    r_score = rouge.compute()
    print(r_score)
def is_smile_valid(smile):
    molecule = Chem.MolFromSmiles(smile)
    return molecule is not None

# is_smile_valid("CC(=O)O[C@@H]1C[C@@H]2[C@](C=CC(=O)C2(C)C)([C@@H]3[C@@]1(C4=CC(=O)[C@H]([C@@]4(CC3)C)C5=COC=C5)C)C")
# with open("test_text2SMILES_I2V_gio_method_base_correct_format.txt", "r") as file:
#     for line in file:
#         split = line.split("\t")
#         pred = split[0].split(": ")[1]
#         true = split[1].strip()
#         if is_smile_valid(pred):
#             pass
#         else:
#             print(f"invalid - pred {pred} => True {true}")
#
#         if is_smile_valid(true):
#             pass
#         else:
#             print("invalid - true")


# model = AutoModelForSeq2SeqLM.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
# tokenizer = AutoTokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
#
#
# max_length = 512
# num_beams = 10
# batch_size = 3
#
# input_batch, target_batch = [], []
# total_count = 0
#
# with open("dataset/Text2SMILES_Gio/test.txt", "r") as infile:
#     for i, line in enumerate(infile):
#         split = line.split("\t")
#         input_batch.append(split[0])
#         target_batch.append(split[1].strip())
#
#         if len(input_batch) == batch_size:
#             total_count += process_batch(input_batch, target_batch, model, tokenizer, max_length, num_beams)
#             input_batch, target_batch = [], []  # Reset the batch
#
#     # Process the remaining batch
#     if input_batch:
#         total_count += process_batch(input_batch, target_batch, model, tokenizer, max_length, num_beams)
#
# # print("Total correct predictions:", total_count, "Total processed:", i + 1)
# import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# #
# # Check if CUDA is available
#
# max_length = 512
# num_beams = 3
#
# # rouge = ROUGEScore()
#
# model_path = "model_241123_text2SMILES_I2V_3.pt"
# model = torch.load(model_path)
# tokenizer = AutoTokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)
#
#
# # Move model to the selected device (GPU or CPU)
#
# count = 0
# with open("test_text2SMILES_I2V_gio_method_base_correct_format.txt", "r") as infile:
#     for i, line in enumerate(infile):
#         split = line.split("\t")
#         input_text = split[0]
#         target = split[1].strip()
#         text = tokenizer(input_text, return_tensors="pt")
#
#         # Move tensors to the same device as model
#         text = {k: v.to(device) for k, v in text.items()}
#
#         output = model.generate(**text, max_length=max_length, num_beams=num_beams)
#         output = tokenizer.decode(output[0].cpu())
#
#         output = output.split(tokenizer.eos_token)[0]
#         output = output.replace(tokenizer.pad_token,"")
#         output = output.replace("<unk>","\\\\")
#         output = output.strip()
#
#         pred_canonical = canonical_smiles(output)
#         true_canonical = canonical_smiles(target)
#         if true_canonical == pred_canonical:
#             count += 1
#             print("correct")
#         with open("test_text2SMILES_I2V_num_beam_3_011223.txt", "a") as file:
#             print(f"Pred:\t{output}\tTrue:\t{target}", file=file)
#         print(i)
# print("count", count, "total", i, "acc", count/i)






with open("test_text2SMILES_I2V_gio_method_base_correct_format.txt", "r") as file:
    count = 0
    total = 0
    for line in file:
        split = line.split("\t")
        pred = split[0].split(": ")[1]
        true = split[1].strip()

        pred_canonical = canonical_smiles(pred)
        true_canonical = canonical_smiles(true)


        total += 1
        if pred_canonical == true_canonical:
            count += 1

    accuracy = count / total
    print("Acc =", accuracy)

# ### new preds format for molecule validation
# with open("test_text2SMILES_I2V_gio_method_for_pred_2.txt", "r") as file:
#     count = 0
#     total = 0
#     for line in file:
#
#         split = line.split("\t")
#         if split[0][-1] == "3":
#             pass
#
#         pred = split[1].strip()
#         true = split[3].strip()
#
#         pred_canonical = canonical_smiles(pred)
#         true_canonical = canonical_smiles(true)
#
#         total += 1
#         if pred_canonical == true_canonical:
#             count += 1
#
#     accuracy = count / total if total > 0 else 0
#     print("Acc =", accuracy)


# ### Convert dataset from Gio format to my format
# with open("dataset/Text2SMILES_Gio/Original_format/train.txt", "r") as infile:
#     with open("dataset/Text2SMILES_Gio/train.txt", "w") as outfile:
#         for i, line in enumerate(infile):
#             if i == 0:
#                 continue
#             split = line.split("\t")
#             print("Text2SMILES: ", split[2].strip(), "\t", split[1], file=outfile)




### Creation of i2v dataset

# with open("dataset/train_i2v_BGC2SMM_251023.txt", "a") as f:
#     with open("predictions_train_230923_pfam_BGC2SMM_v4.txt", "r") as infile:
#         for line in infile:
#             split = line.split("\t")
#             if split[0] == "Epoch 16/18":
#                 true = split[1].split("True: ")[1][1:-1].split(", ")
#                 pred = split[2].split("Pred: ")[1][1:-1].split(", ")
#                 for p, t in zip(pred, true):
#                     p = p.replace("'", "")
#                     t = t.replace("'", "")
#                     print(f"invalid2validSMILE: {p}\t{t}", file=f)

### check for data leakage between test and train set (only target)
# def get_set(file_path, target):
#     with open (file_path, "r") as f:
#         if target:
#             target_set = set()
#             for line in f:
#                 # find when a line already is in the set
#                 label = line.split("\t")[1]
#                 if label in target_set:
#                     # print(label)
#                     pass
#                 else:
#                     target_set.add(label)
#             return target_set
#         else:
#             target_set=set()
#             for line in f:
#                 # find when a line already is in the set
#                 if line in target_set:
#                     # print(line, "Full line")
#                     pass
#                 else:
#                     target_set.add(line)
#             return target_set
#
# test = get_set("dataset/pfam2SMILES/test_pfam_v2.txt", False)
# train = get_set("dataset/pfam2SMILES/train_pfam_v2.txt", False)
# # datset = get_set("dataset/pfam2SMILES/dataset_pfam2SMILES_v2.txt", True)
# # find intersection
# print(len(test))
# print(len(train))
# intersection = test.intersection(train)
# print(len(intersection), "inter")
# # print(intersection)

# ### plot for distrubtion of char error
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Reading the file and populating the DataFrame
# file_path = 'predictions_test_150723_i2v_pfam_correct_dataset_v1.txt'  # Replace with your actual file path
# epochs = []
# true_labels = []
# predictions = []
# tasks = []
#
# with open(file_path, 'r') as f:
#     for line in f:
#         parts = line.strip().split("\t")
#         if len(parts) == 5:  # Adjusted condition based on your file format
#             epochs.append(parts[0].split(" ")[1])
#             true_labels.append(parts[1].split("True: ")[1])
#             predictions.append(parts[2].split("Pred: ")[1])
#             tasks.append(parts[4])  # Adjusted index
#
# df = pd.DataFrame({
#     'Epoch': epochs,
#     'True_Label': true_labels,
#     'Prediction': predictions,
#     'Task': tasks
# })
#
# # Filter the DataFrame for the 15th epoch
# df_15_epoch = df[df['Epoch'] == '15/18']
#
# # Initialize an empty list to store the number of wrong characters for each sorted prediction
# wrong_char_counts_sorted_diff_len = []
#
# # Calculate the number of wrong characters for each sorted prediction compared to the sorted true label
# for true_label, prediction in zip(df_15_epoch['True_Label'], df_15_epoch['Prediction']):
#     # Remove quotes and brackets to compare the actual strings
#     true_label = true_label[2:-2]
#     prediction = prediction[2:-2]
#
#     # Sort the strings
#     true_label_sorted = ''.join(sorted(true_label))
#     prediction_sorted = ''.join(sorted(prediction))
#
#     # Initialize count for this pair
#     wrong_count = 0
#
#     # Compare each character up to the length of the shorter string
#     for t_char, p_char in zip(true_label_sorted, prediction_sorted):
#         if t_char != p_char:
#             wrong_count += 1
#
#     # Add the difference in lengths to the wrong_count (if any)
#     wrong_count += abs(len(true_label_sorted) - len(prediction_sorted))
#
#     # Append the count to the list
#     wrong_char_counts_sorted_diff_len.append(wrong_count)
#
# # Plot the histogram for sorted strings, accounting for different lengths
# plt.hist(wrong_char_counts_sorted_diff_len, bins=20, edgecolor='black')
# plt.xlabel('Number of Wrong Characters')
# plt.ylabel('Frequency')
# plt.title('Distribution of Wrong Characters Pfam I2V')
# plt.show()
#
# ### with densities
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from scipy.stats import norm
#
# # Reading the file and populating the DataFrame
# file_path = 'predictions_test_150723_i2v_pfam_correct_dataset_v1.txt'  # Replace with your actual file path
# epochs = []
# true_labels = []
# predictions = []
# tasks = []
#
# with open(file_path, 'r') as f:
#     for line in f:
#         parts = line.strip().split("\t")
#         if len(parts) == 5:  # Adjusted condition based on your file format
#             epochs.append(parts[0].split(" ")[1])
#             true_labels.append(parts[1].split("True: ")[1])
#             predictions.append(parts[2].split("Pred: ")[1])
#             tasks.append(parts[4])
#
# df = pd.DataFrame({
#     'Epoch': epochs,
#     'True_Label': true_labels,
#     'Prediction': predictions,
#     'Task': tasks
# })
#
# # Filter the DataFrame for the 15th epoch
# df_15_epoch = df[df['Epoch'] == '15/18']
#
# # Initialize an empty list to store the number of wrong characters for each sorted prediction
# wrong_char_counts_sorted_diff_len = []
#
# # Calculate the number of wrong characters for each sorted prediction compared to the sorted true label
# for true_label, prediction in zip(df_15_epoch['True_Label'], df_15_epoch['Prediction']):
#     true_label = true_label[2:-2]
#     prediction = prediction[2:-2]
#     true_label_sorted = ''.join(sorted(true_label))
#     prediction_sorted = ''.join(sorted(prediction))
#     wrong_count = 0
#     for t_char, p_char in zip(true_label_sorted, prediction_sorted):
#         if t_char != p_char:
#             wrong_count += 1
#     wrong_count += abs(len(true_label_sorted) - len(prediction_sorted))
#     wrong_char_counts_sorted_diff_len.append(wrong_count)
#
# # Set up the matplotlib figure
# plt.figure(figsize=(12, 6))
#
# # Plot histogram using matplotlib (normalized)
# plt.hist(wrong_char_counts_sorted_diff_len, bins=20, edgecolor='black', alpha=0.5, density=True, label='Histogram')
#
# # Plot density using seaborn for KDE
# sns.kdeplot(wrong_char_counts_sorted_diff_len, bw_adjust=0.5, fill=True, color='blue', label='KDE Density')
#
# # Fit a normal distribution to the data
# mu, std = norm.fit(wrong_char_counts_sorted_diff_len)
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2, label='Normal Density')
#
# # Add labels and title
# plt.xlabel('Number of Wrong Characters')
# plt.ylabel('Density')
# plt.title('Distribution of Wrong Characters Pfam I2V')
# plt.legend()
# plt.show()


### Density plots.

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from scipy.stats import norm
#
#
# # Function to read the file and return a list of wrong character counts for the 15th epoch
# def read_and_process(file_path, ful):
#     epochs = []
#     true_labels = []
#     predictions = []
#     tasks = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split("\t")
#             if len(parts) == 5:
#                 epochs.append(parts[0].split(" ")[1])
#                 true_labels.append(parts[1].split("True: ")[1])
#                 predictions.append(parts[2].split("Pred: ")[1])
#                 tasks.append(parts[4])
#     df = pd.DataFrame({
#         'Epoch': epochs,
#         'True_Label': true_labels,
#         'Prediction': predictions,
#         'Task': tasks
#     })
#     df_15_epoch = df[df['Epoch'] == f'15/{ful}']
#     wrong_char_counts_sorted_diff_len = []
#     for true_label, prediction in zip(df_15_epoch['True_Label'], df_15_epoch['Prediction']):
#         true_label = true_label[2:-2]
#         prediction = prediction[2:-2]
#         true_label_sorted = ''.join(sorted(true_label))
#         prediction_sorted = ''.join(sorted(prediction))
#         wrong_count = 0
#         for t_char, p_char in zip(true_label_sorted, prediction_sorted):
#             if t_char != p_char:
#                 wrong_count += 1
#         wrong_count += abs(len(true_label_sorted) - len(prediction_sorted))
#         wrong_char_counts_sorted_diff_len.append(wrong_count)
#     return wrong_char_counts_sorted_diff_len
#
#
# # File paths
# file_path1 = 'predictions_test_150723_i2v_pfam_correct_dataset_v1.txt'  # Replace with your actual file path
# file_path2 = 'predictions_test_110923_pfam.txt'  # Replace with your actual file path
#
# # Read and process both files
# wrong_char_counts1 = read_and_process(file_path1, "18")
# wrong_char_counts2 = read_and_process(file_path2, "20")
#
# # Set up the matplotlib figure
# plt.figure(figsize=(12, 6))
# plt.rcParams.update({'font.size': 14})
#
# # Check if the wrong_char_counts for both datasets are non-empty to avoid warnings and errors
# if wrong_char_counts1 and wrong_char_counts2:
#     # Plot histogram using matplotlib (normalized) for both datasets
#     # plt.hist(wrong_char_counts1, bins=20, color='red', edgecolor='black', alpha=0.5, density=True,
#     #          label='Dataset 1 - Histogram')
#     # plt.hist(wrong_char_counts2, bins=20, color='orange', edgecolor='black', alpha=0.5, density=True,
#     #          label='Dataset 2 - Histogram')
#
#     # Plot density using seaborn for KDE for both datasets
#     sns.kdeplot(wrong_char_counts1, fill=True, bw_adjust=0.5, color='blue', label='w/ Invalid2Valid')
#     sns.kdeplot(wrong_char_counts2, fill=True, bw_adjust=0.5, color='green', label='w/o Invalid2Valid')
#
#     # Fit a normal distribution to the data for both datasets
#     mu1, std1 = norm.fit(wrong_char_counts1)
#     mu2, std2 = norm.fit(wrong_char_counts2)
#
#     # xmin, xmax = plt.xlim()
#     # x = np.linspace(xmin, xmax, 100)
#     # p1 = norm.pdf(x, mu1, std1)
#     # p2 = norm.pdf(x, mu2, std2)
#     #
#     # plt.plot(x, p1, 'k', linewidth=2, label='I2V - Normal Density')
#     # plt.plot(x, p2, 'm', linewidth=2, label='pfam-only - Normal Density')
# else:
#     plt.text(0.5, 0.5, 'One or both datasets have no data for the 15th epoch.', horizontalalignment='center',
#              verticalalignment='center', fontsize=12)
#
# # Add labels and title
# plt.xlabel('Number of Wrong Characters', fontsize=16)
# plt.ylabel('Density', fontsize=16)
# # plt.title('Distribution and Density of Wrong Characters in both pfam-only and I2V', fontsize=18)
# plt.legend(fontsize=14)
# plt.xlim(0, 100)
# plt.savefig('wrong_chars_density_i2v_pfam.pdf', format='pdf', dpi=2000)
# plt.show()

### plots for scores new format
# import pandas as pd
# import matplotlib.pyplot as plt
#
# def process_lines(lines):
#     data = []
#     for line in lines:
#         split_line = line.split("\t")
#         epoch = split_line[0][6:-3]
#         accuracy = split_line[1].split(": ")[1]
#         avg_rouge_f1 = split_line[3]
#         avg_char_error_rate = split_line[7]
#         avg_sacrebleu_score = split_line[9]
#         num_correct_val_mols = split_line[10].split(": ")[1].rstrip()
#
#         data.append({
#             "Epoch": epoch,
#             "Accuracy": accuracy,
#             "ROUGE-F1": avg_rouge_f1,
#             "Char Error Rate": avg_char_error_rate,
#             "Avg SacreBLEU Score": avg_sacrebleu_score,
#             "Num correct val mols": num_correct_val_mols,
#         })
#     return data
#
# # Load files and split them into lines
# with open("scores_130723_pfam2SMILES_v2.txt", "r") as file:
#     lines_v3 = file.readlines()
#
# with open("scores_150723_i2v_pfam_correct_dataset_v1.txt", "r") as file:
#     lines_v5 = file.readlines()
#
# # Process lines for both files
# train_lines_v3, test_lines_v3 = lines_v3[::2], lines_v3[1::2]
# train_data_v3, test_data_v3 = process_lines(train_lines_v3), process_lines(test_lines_v3)
# train_df_v3, test_df_v3 = pd.DataFrame(train_data_v3).apply(pd.to_numeric, errors='ignore'), pd.DataFrame(test_data_v3).apply(pd.to_numeric, errors='ignore')
#
# train_lines_v5, test_lines_v5 = lines_v5[::2], lines_v5[1::2]
# train_data_v5, test_data_v5 = process_lines(train_lines_v5), process_lines(test_lines_v5)
# train_df_v5, test_df_v5 = pd.DataFrame(train_data_v5).apply(pd.to_numeric, errors='ignore'), pd.DataFrame(test_data_v5).apply(pd.to_numeric, errors='ignore')
#
# # Normalization
# train_df_v3['Num correct val mols'] /= 1191
# test_df_v3['Num correct val mols'] /= 298
# train_df_v5['Num correct val mols'] /= 4609
# test_df_v5['Num correct val mols'] /= 898
#
# # Create the subplots
# fig, ax = plt.subplots(2, 2, figsize=(15, 10))
#
# # Adjust global font size
# plt.rcParams.update({'font.size': 14})
#
# # Create subplots for all metrics
# for i, metric in enumerate(['Accuracy', 'ROUGE-F1', 'Char Error Rate', 'Num correct val mols']):
#     row, col = divmod(i, 2)
#     ax[row, col].plot(train_df_v3['Epoch'], train_df_v3[metric], ls=':', alpha=0.5,
#                       color='green')
#     ax[row, col].plot(test_df_v3['Epoch'], test_df_v3[metric], label=f'w/o Invalid2Valid', color='green')
#     ax[row, col].plot(train_df_v5['Epoch'], train_df_v5[metric], ls=':', alpha=0.5, color='blue')
#     ax[row, col].plot(test_df_v5['Epoch'], test_df_v5[metric], label=f'w/ Invalid2Valid', color='blue')
#
#     # ax[row, col].set_title(f'Epoch vs {metric}', fontsize=16)
#     ax[row, col].set_xlabel('Epoch', fontsize=16)
#     ax[row, col].set_ylabel(metric, fontsize=16)
#     ax[row, col].legend(fontsize=14)
# # save
# plt.tight_layout()
# plt.savefig('scores_i2v_vs_pfam_.pdf', format='pdf', dpi=2000)
# plt.show()

### Plot for SMILES2Activity

# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# def process_lines(lines):
#     data = []
#     for line in lines:
#         split_line = line.split("\t")
#         epoch = split_line[0][6:-3]
#         accuracy = split_line[1].split(": ")[1]
#         avg_rouge_f1 = split_line[3]
#         avg_char_error_rate = split_line[7]
#         avg_sacrebleu_score = split_line[9]
#         num_correct_val_mols = split_line[10].split(": ")[1].rstrip()
#
#         data.append({
#             "Epoch": epoch,
#             "Accuracy": accuracy,
#             "ROUGE-F1": avg_rouge_f1,
#             "Char Error Rate": avg_char_error_rate,
#             "Avg SacreBLEU Score": avg_sacrebleu_score,
#             "Num correct val mols": num_correct_val_mols,
#         })
#     return data
#
#
# # Load files and split them into lines
# with open("scores_150723_grammarcheck_activities_v1.txt", "r") as file:
#     lines_v1 = file.readlines()
#
# with open("scores_150723_smiles2act_removed_single_labels_v2.txt", "r") as file:
#     lines_v2 = file.readlines()
#
# # Process lines for both files
# train_lines_v1, test_lines_v1 = lines_v1[::2], lines_v1[1::2]
# train_data_v1, test_data_v1 = process_lines(train_lines_v1), process_lines(test_lines_v1)
# train_df_v1, test_df_v1 = pd.DataFrame(train_data_v1).apply(pd.to_numeric, errors='ignore'), pd.DataFrame(
#     test_data_v1).apply(pd.to_numeric, errors='ignore')
#
# train_lines_v2, test_lines_v2 = lines_v2[::2], lines_v2[1::2]
# train_data_v2, test_data_v2 = process_lines(train_lines_v2), process_lines(test_lines_v2)
# train_df_v2, test_df_v2 = pd.DataFrame(train_data_v2).apply(pd.to_numeric, errors='ignore'), pd.DataFrame(
#     test_data_v2).apply(pd.to_numeric, errors='ignore')
#
# # Normalization (Note: Update normalization factors based on your data)
# train_df_v1['Num correct val mols'] /= 1  # Update this
# test_df_v1['Num correct val mols'] /= 1  # Update this
# train_df_v2['Num correct val mols'] /= 1  # Update this
# test_df_v2['Num correct val mols'] /= 1  # Update this
#
# # Create the subplots focusing only on 'Accuracy' and 'ROUGE-F1'
# fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#
# # Adjust global font size
# plt.rcParams.update({'font.size': 14})
#
# # Create subplots for selected metrics ('Accuracy' and 'ROUGE-F1')
# for i, metric in enumerate(['Accuracy', 'ROUGE-F1']):
#     ax[i].plot(train_df_v1['Epoch'], train_df_v1[metric], ls=':', alpha=0.5, color='blue')
#     ax[i].plot(test_df_v1['Epoch'], test_df_v1[metric], label='w/ Invalid2Valid', color='blue')
#     ax[i].plot(train_df_v2['Epoch'], train_df_v2[metric], ls=':', alpha=0.5, color='green')
#     ax[i].plot(test_df_v2['Epoch'], test_df_v2[metric], label='w/o Invalid2Valid', color='green')
#
#     ax[i].set_xlabel('Epoch', fontsize=16)
#     ax[i].set_ylabel(metric, fontsize=16)
#     ax[i].legend(fontsize=14)
#
# # Save and show the updated plot
# plt.tight_layout()
# plt.savefig("scores_SMILES2acts.pdf", format='pdf', dpi=2000)
# plt.show()

### plots for scores old format
#
# import matplotlib.pyplot as plt
#
# # List of file names
# files = ["scores_140723_pfam_invalid2valid_v2.txt", "scores_130723_pfam2SMILES_v2.txt"]
#
# # Initialize lists to store the extracted data
# data = {}
# for file in files:
#     data[file] = {
#         "epochs_train": [],
#         "epochs_test": [],
#         "rouge1_train": [],
#         "rouge1_test": [],
#         "bleu_train": [],
#         "bleu_test": [],
#         "char_error_rate_train": [],
#         "char_error_rate_test": [],
#         "sacrebleu_train": [],
#         "sacrebleu_test": [],
#         "num_correct_val_mols_train": [],
#         "num_correct_val_mols_test": [],
#     }
#
#     train_flag = False
#
#     # Read and process data from the file
#     with open(file, "r") as f:
#         for line in f.readlines():
#             if line.startswith("Epoch"):
#                 values = line.split('\t')
#                 epoch_num = int(values[0].split()[1].split('/')[0])
#                 if epoch_num < 0:
#                     continue  # Skip the first two epochs
#
#                 if train_flag:
#                     data[file]["epochs_test"].append(epoch_num)
#                     data[file]["rouge1_test"].append(float(values[2]))
#                     data[file]["bleu_test"].append(float(values[4]))
#                     data[file]["char_error_rate_test"].append(float(values[6]))
#                     data[file]["sacrebleu_test"].append(float(values[8]))
#                     data[file]["num_correct_val_mols_test"].append(int(values[9].split(':')[1][1:]))
#                     train_flag = False
#
#                 elif not train_flag:
#                     data[file]["epochs_train"].append(epoch_num)
#                     data[file]["rouge1_train"].append(float(values[2]))
#                     data[file]["bleu_train"].append(float(values[4]))
#                     data[file]["char_error_rate_train"].append(float(values[6]))
#                     data[file]["sacrebleu_train"].append(float(values[8]))
#                     data[file]["num_correct_val_mols_train"].append(int(values[9].split(':')[1][1:]))
#                     train_flag = True
#
# # Prepare metric names
# metric_names = ["rouge1", "char_error_rate", "sacrebleu", "num_correct_val_mols"]
#
# def plot_metrics(data, metric_names):
#     plt.figure(figsize=(12, 10))
#
#     legend_labels = {
#         "scores_160523_GT4SD_multitask-text-and-chemistry-t5-small-augm.txt": "ESM2-chemistry-t5-small-augm",
#         "scores_070523_molt5_base.txt": "ESM2-molt5_base"
#     }
#
#     for i, metric_name in enumerate(metric_names):
#         for file in data.keys():
#             metric_name_key = metric_name.replace('-', '_')
#             plt.subplot(2, 2, i+1)
#             # plt.plot(data[file]["epochs_train"], data[file][f"{metric_name_key}_train"], label=f'Train {file}')
#             plt.plot(data[file]["epochs_test"], data[file][f"{metric_name_key}_test"], label=f'Test {legend_labels.get(file, file)}')
#             plt.xlabel("Epoch")
#             plt.ylabel(metric_name)
#             plt.title(f"{metric_name} vs. Epoch")
#             plt.legend()
#
#     plt.tight_layout()
#     plt.suptitle("ESM_Chem T5 vs ESM2-T5 - Test data", fontsize=16, y=1.05)
#     plt.savefig('Chem_T5_esm_vs_ESM2_T5_test_data.pdf', bbox_inches='tight', dpi=300, format='pdf')
#
#     plt.show()
#
#
#
# # Create a single plot with 4 subplots arranged in a 2x2 grid
# plot_metrics(data, metric_names)

# from Bio.Blast import NCBIWWW
# from Bio import SeqIO
#
#
# import time
# from concurrent.futures import ThreadPoolExecutor
# from io import StringIO
# from Bio import SeqIO
# from Bio.Blast import NCBIWWW
#
# s = time.time()
#
# def search_homologous_sequences(args):
#     fasta_string, outname = args
#     if int(outname.split('_')[3]) >= 850:
#         database="nr"
#         e_value=0.01
#         fasta_string_with_header = f">seq\n{fasta_string}"
#         fasta_io = StringIO(fasta_string_with_header)
#         try:
#             record = SeqIO.read(fasta_io, format="fasta")
#             result_handle = NCBIWWW.qblast("blastp", database, record.seq, expect=e_value)
#             with open(f"blastp_temp_files/{outname}.txt", "w") as out_handle:
#                 out_handle.write(result_handle.read())
#                 print(out_handle)
#             result_handle.close()
#         except Exception as e:
#             print(f"An error occurred: {e}")
#             return

# fasta_strings = []
# outnames = []
#
# with open("dataset/protein_SMILE/dataset_protein_peptides_complete_v2.txt", "r") as f:
#     for i, line in enumerate(f):
#         line = line.split('\t')[0]
#         data = line.split('_')[1:]
#
#         for j, seq in enumerate(data):
#             len_ = len(seq)
#             if len_ >= 850:
#                 fasta_strings.append(seq)
#                 outnames.append(f"seq_{i}_{j}_{len_}")
# print(len(fasta_strings))
#
# args = zip(fasta_strings, outnames)

# num_workers = 10
#
# with ThreadPoolExecutor(max_workers=num_workers) as executor:
#     executor.map(search_homologous_sequences, args)
# e = time.time()
# print(e-s, "seconds")
#
# import re
# import os
# from Bio.Align import AlignInfo
# from Bio import AlignIO
# import math
#
# def contains_gap(filename):
#     with open(filename, "r") as f:
#         for line in f:
#             if "-" in line:
#                 return True
#     return False
#
#
# def shannon_entropy(list_input):
#     unique_base = set(list_input)
#     entropy = 0
#     for base in unique_base:
#         p_x = list_input.count(base) / len(list_input)
#         if p_x > 0:
#             entropy += - p_x * math.log2(p_x)
#     return entropy
#
# def process_files(start, end):
#     file_list = sorted(os.listdir("blast/"))
#     file_list = file_list[start:end]
#
#     for ii, filename in enumerate(file_list, start=start):
#
#         file_identifier = filename.split('_')[2:]
#         file_identifier[-1] = file_identifier[-1].split('.')[0]
#         print(ii)
#         print(file_identifier)
#
#
#
#         with open(f"blast/{filename}", "r") as f:
#             query_list = []
#             subject_list = []
#             q_string, s_string = "", ""
#             for line in f:
#                 if line.startswith('>'):
#                     if q_string and s_string:
#                         if len(query_list) == 0:
#                             query_list.append(re.sub(r'\s', '', q_string))
#                         subject_list.append(re.sub(r'\s', '', s_string))
#                     q_string, s_string = "", ""
#
#                 if line.startswith('Query '):
#                     query = line[10:].strip()
#                     query = ''.join([i for i in query if not i.isdigit()])
#                     q_string += query
#
#                 elif line.startswith('Sbjct '):
#                     subject = line[10:].strip()
#                     subject = ''.join([i for i in subject if not i.isdigit()])
#                     s_string += subject
#
#
#             if q_string and s_string:
#                 if len(query_list) == 0:
#                     query_list.append(re.sub(r'\s', '', q_string))
#                 subject_list.append(re.sub(r'\s', '', s_string))
#
#             subject_list.append(query_list[0])
#
#
#
#             with open("input_sequences.txt", "w") as f:
#                 for i, seq in enumerate(subject_list):
#                     f.write(f">seq_{i}\n{seq}\n")
#
#
#
#             # Define the paths and execute the Clustal Omega command
#             file_path = os.getcwd()
#             data_path = os.getcwd()
#             file = 'input_sequences.txt'
#             os.system(f"clustalo -i {file_path}/{file} --dealign -o {data_path}/{file[:-4]}.fasta --force --threads=10")
#
#             # Now read the alignment after the appropriate alignment has been done
#             alignment = AlignIO.read("input_sequences.fasta", "fasta")
#             summary_align = AlignInfo.SummaryInfo(alignment)
#
#             scores = []
#             for i in range(len(alignment[0])):
#                 column_bases = alignment[:, i]
#                 scores.append(shannon_entropy(column_bases))
#
#             # Use the dumb_consensus method to get the consensus sequence
#             consensus = summary_align.dumb_consensus()
#
#
#             while len(consensus) > 850:
#                 index = scores.index(max(scores))
#                 consensus = consensus[:index] + consensus[index + 1:]
#                 del scores[index]
#
#             shorten = consensus
#
#
#             # First read all lines from the file and close it
#             with open("Transformer_DB_Curation_MIBiG/code/dataset/protein_SMILE/dataset_protein_peptides_complete_v3_shorten.txt", "r") as infile:
#                 lines = infile.readlines()
#
#             # Now open the file in write mode to write back the modified lines
#             with open("Transformer_DB_Curation_MIBiG/code/dataset/protein_SMILE/dataset_protein_peptides_complete_v3_shorten.txt", "w") as outfile:
#                 for i, line in enumerate(lines):
#                     d1 = line.split('\t')[0]
#                     smile = line.split('\t')[1]
#                     data = d1.split('_')[1:]
#
#                     if i == int(file_identifier[0]):
#
#                         # Check - To insure correct file
#                         if len(data[int(file_identifier[1])]) == int(file_identifier[2]):
#                             print("replaced it")
#                             data[int(file_identifier[1])] = shorten
#                             print(type(d1), type(data))
#                             # Convert the Seq object to a string
#                             data = [str(element) for element in data]
#                             # Recreate the string for the line
#                             new_d1 = '_'.join([d1.split('_')[0]] + data)
#                             line = '\t'.join([new_d1, smile])
#
#                     outfile.write(line)
#
# import argparse
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("start", help="Start index for file processing", type=int)
#     parser.add_argument("end", help="End index for file processing", type=int)
#     args = parser.parse_args()
#
#     process_files(args.start, args.end)
#
# c = 0
# with open("dataset_protein_peptides_complete_v3_shorten_0.txt", "r") as f:
#     for i, line in enumerate(f):
#         splir = line.split('\t')
#         data = splir[0].split('_')
#
#         # check if length is under 850 for all elements in data
#         for j in range(1, len(data)):
#             if len(data[j]) >   850:
#                 print(i, j, len(data[j]))
#                 c += 1
# print(c)

# import re
# import os
# import argparse
# from Bio.Align import AlignInfo
# from Bio import AlignIO
# import math
#
# def shannon_entropy(list_input):
#     unique_base = set(list_input)
#     entropy = 0
#     for base in unique_base:
#         p_x = list_input.count(base) / len(list_input)
#         if p_x > 0:
#             entropy += - p_x * math.log2(p_x)
#     return entropy
#
# def process_files(start, end, job_id):
#     # Read the sorted list of files from the stored list
#     with open("file_order.txt", "r") as f:
#         file_list = [line.strip() for line in f]
#
#     file_list = file_list[start:end]
#
#     print("start")
#     for ii, filename in enumerate(file_list, start=start):
#         with open(f"blast_rest/{filename}", "r") as f:
#             query_list = []
#             subject_list = []
#             q_string, s_string = "", ""
#             for line in f:
#                 if line.startswith('>'):
#                     if q_string and s_string:
#                         if len(query_list) == 0:
#                             query_list.append(re.sub(r'\s', '', q_string))
#                         subject_list.append(re.sub(r'\s', '', s_string))
#                     q_string, s_string = "", ""
#
#                 if line.startswith('Query '):
#                     query = line[10:].strip()
#                     query = ''.join([i for i in query if not i.isdigit()])
#                     q_string += query
#
#                 elif line.startswith('Sbjct '):
#                     subject = line[10:].strip()
#                     subject = ''.join([i for i in subject if not i.isdigit()])
#                     s_string += subject
#
#             if q_string and s_string:
#                 if len(query_list) == 0:
#                     query_list.append(re.sub(r'\s', '', q_string))
#                 subject_list.append(re.sub(r'\s', '', s_string))
#
#             subject_list.append(query_list[0])
#
#             with open(f"input_sequences_{job_id}.txt", "w") as f:
#                 for i, seq in enumerate(subject_list):
#                     f.write(f">seq_{i}\n{seq}\n")
#             file_path = os.getcwd()
#             data_path = os.getcwd()
#             file = f'input_sequences_{job_id}.txt'
#             os.system(f"clustalo -i {file_path}/{file} --dealign -o {data_path}/{file[:-4]}.fasta --force --threads=10")
#             alignment = AlignIO.read(f"input_sequences_{job_id}.fasta", "fasta")
#             summary_align = AlignInfo.SummaryInfo(alignment)
#             scores = []
#             for i in range(len(alignment[0])):
#                 column_bases = alignment[:, i]
#                 scores.append(shannon_entropy(column_bases))
#
#             consensus = summary_align.dumb_consensus()
#
#             while len(consensus) > 850:
#                 index = scores.index(max(scores))
#                 consensus = consensus[:index] + consensus[index + 1:]
#                 del scores[index]
#
#             shorten = consensus
#             print(len(shorten))
#             with open(f"shorten/shorten_{filename}", "w") as out:
#                 print(shorten , file=out, end="")
#



# if __name__ == "__main__":
#     # Write the sorted list of files to a file
#     file_list = sorted(os.listdir("blast_rest/"))
#     with open("file_order.txt", "w") as f:
#         for filename in file_list:
#             f.write(filename + '\n')
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("start", help="Start index for file processing", type=int)
#     parser.add_argument("end", help="End index for file processing", type=int)
#     parser.add_argument("job_id", help="Job ID for this task", type=int)
#     args = parser.parse_args()
#
#     process_files(args.start, args.end, args.job_id)


# import os
#
# # Read the main file
# with open("dataset/protein_SMILE/dataset_protein_peptides_complete_v3_shorten.txt", 'r') as f:
#     data = f.readlines()
#
# # Loop through all the files in the directory
# for i, file_name in enumerate (os.listdir('../shorten_v2/')):
#     if file_name.endswith('.txt'):
#         # Extract line number, position, and length from the file name
#         line_number = int(file_name.split('_')[3])
#         if file_name.split('_')[4] == '0':
#             position = 0
#         elif file_name.split('_')[4] == '00000':
#             position = 1
#         else:
#             position = int(file_name.split('_')[4])+1
#
#         length = int(file_name.split('_')[5].split('.')[0])
#
#         print(file_name)
#
#         # Read the sequence from the file
#         with open(f'../shorten_v2/{file_name}', 'r') as f:
#             new_sequence = f.read().strip()
#
#         # Get the line from the main file
#
#         print(line_number, 'pos: ', position, 'len: ', length)
#         line = data[line_number]
#
#
#         # Split the line
#         line_parts = line.split('\t')[0].split(': ')[1].split('_')
#         smile = line.split('\t')[1]
#
#         # Get the original sequence
#         original_sequence = line_parts[position]
#
#         print(len(original_sequence), len(new_sequence))
#
#
#         # Make sure the original sequence has the correct length
#         if len(original_sequence) == length:
#             print("Replacing sequence")
#             # Replace the original sequence with the new one
#             line_parts[position] = new_sequence
#
#             # Combine the line parts back into a string
#             new_line = '_'.join(line_parts)
#
#             # add ProteinSeqs2SMILE: to the beginning of the line
#             new_line = 'ProteinSeqsShort2SMILE: ' + new_line + '\t' + smile
#
#             # Replace the line in the main data
#             data[line_number] = new_line
#         else:
#             print("Lengths don't match")
#             break
#
#
# # Write the modified data back to the main file
# with open("dataset/protein_SMILE/dataset_protein_peptides_complete_v3_3_shorten.txt", 'w') as f:
#     f.writelines(data)


# with open("dataset/protein_SMILE/dataset_protein_peptides_complete_v3_3_shorten.txt", 'r') as f:
#     data = f.readlines()
#     c = 0
#     for i, line in enumerate(data):
#         line_parts = line.split('\t')[0].split(': ')[1].split('_')
#         for j, item in enumerate(line_parts):
#             if len(item) > 850:
#                 print(i, j, len(item))
#                 c += 1
#     print(c)


# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel, AdamW
# import time
# from sklearn.metrics import accuracy_score, f1_score
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import argparse as arg
# from torchmetrics.text import BLEUScore, ROUGEScore
# from torchmetrics import CharErrorRate, SacreBLEUScore
# from rdkit import Chem
#
# def is_valid_smiles(smiles: str) -> bool:
#     mol = Chem.MolFromSmiles(smiles)
#     return mol is not None
#
# parser = arg.ArgumentParser()
# parser.add_argument("-o", "--output_file_name", type=str, default="unknown", )
# args = parser.parse_args()
#
#
# class Dataset(Dataset):
#     def __init__(self, file_path, tokenizer, max_length=851):
#         self.file_path = file_path
#         self.data = self.load_data()
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def load_data(self):
#         data = []
#         with open(self.file_path, 'r') as f:
#             for line in f:
#                 text = line.split(': ')[1].split('\t')[0]
#                 label = line.split('\t')[1].strip('\n')
#                 text_list = text.split('_')
#
#                 # Check if any element in text_list is longer than 2000 characters
#                 if all(len(element) <= 851 for element in text_list):
#                     data.append((text_list, label))
#                 else:
#                     truncated_text_list = [element[:851] for element in text_list]
#                     data.append((truncated_text_list, label))
#
#         print(len(data))
#         return data
#
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         text, label = self.data[idx]
#         input_encoding = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding="max_length")
#         target_encoding = self.tokenizer(label, return_tensors="pt", max_length=400, padding="max_length",
#                                          truncation=True)
#
#         return {
#             "input_ids": input_encoding["input_ids"].squeeze(),
#             "attention_mask": input_encoding["attention_mask"].squeeze(),
#             "labels": target_encoding["input_ids"].squeeze(),
#         }
#
#
# start_time = time.time()
#
# # Assume you have a T5 model and tokenizer already
# T5_model_name = 'GT4SD/multitask-text-and-chemistry-t5-base-augm'
# t5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)
# t5_model = torch.load("model_200623_T5_v11_saving_model.pt")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# t5_model.to(device)
#
#
# test_dataset = Dataset("dataset/invalid2validSMILE/test_invalid2validSMILE_ex1.txt", t5_tokenizer)
#
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#
#
# learning_rate = 5e-5
# optimizer = AdamW(list(t5_model.parameters()), lr=learning_rate)
#
# rouge = ROUGEScore()
# bleu = BLEUScore()
# char_error_rate = CharErrorRate()
# sacre_bleu = SacreBLEUScore()
#
# num_epochs = 18
# t5_model.eval()
# # Training loop
# for epoch in range(num_epochs):
#
#     # Similar loop for testing
#
#     rouge_test_accumulated = 0.0
#     bleu_test_accumulated = 0.0
#     char_error_rate_test_accumulated = 0.0
#     sacre_bleu_test_accumulated = 0.0
#     num_test_batches = 0
#     Num_correct_val_mols_test = 0
#     test_outputs = []
#
#     for batch in test_loader:
#         num_test_batches += 1
#         inputs = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)
#
#         with torch.no_grad():
#             outputs = t5_model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
#             test_predicted_labels = t5_tokenizer.decode(outputs.logits[0].argmax(dim=-1).tolist(),
#                                                         skip_special_tokens=True, num_of_beams=10)
#             test_true_labels = [t5_tokenizer.decode(label.tolist(), skip_special_tokens=True) for label in batch["labels"]]
#
#             test_outputs.append({"predicted_label": test_predicted_labels, "true_label": test_true_labels[0]})
#
#             if is_valid_smiles(test_predicted_labels):
#                 Num_correct_val_mols_test += 1
#
#             with open(f"predictions_{args.output_file_name}.txt", "a") as predictions_file:
#                 print(f"Epoch {epoch + 1}/{num_epochs}\tTrue: {test_true_labels}\tPred: {test_predicted_labels}",
#                       file=predictions_file)
#
#             test_rouge_score = rouge(test_predicted_labels, test_true_labels)["rouge1_fmeasure"]
#             test_bleu_score = bleu(test_predicted_labels.split(), [test_true_labels[0].split()])
#             test_char_error_rate_score = char_error_rate(test_predicted_labels, test_true_labels).item()
#             test_sacre_bleu_score = sacre_bleu([test_predicted_labels], [test_true_labels]).item()
#
#             rouge_test_accumulated += test_rouge_score
#             bleu_test_accumulated += test_bleu_score
#             char_error_rate_test_accumulated += test_char_error_rate_score
#             sacre_bleu_test_accumulated += test_sacre_bleu_score
#
#     # Print and save results for this epoch
#     with open(f"scores_{args.output_file_name}.txt", "a") as scores_file:
#
#         print(
#             f"Epoch {epoch + 1}/{num_epochs}\t Avg Test ROUGE-1 F1 Score\t {rouge_test_accumulated / num_test_batches}\tAvg Test BLEU Score\t {bleu_test_accumulated / num_test_batches}\tAvg Test Char Error Rate\t {char_error_rate_test_accumulated / num_test_batches}\tAvg Test SacreBLEU Score\t {sacre_bleu_test_accumulated / num_test_batches}\tNum correct val mols test: {Num_correct_val_mols_test}",
#             file=scores_file)
#     # save the model
#     # if epoch == 17:
#     #     torch.save(t5_model, f"model_{args.output_file_name}.pt")


# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import random
#
# # Email and password (use an app password if 2FA is enabled)
# email = "jakob.larsen949@gmail.com"
# password = "hxtb pryh elfq vtfj "
# # password = "your_app_password"
#
# # Setup the server
# server = smtplib.SMTP('smtp.gmail.com', 587)
# server.starttls()
# server.login(email, password)
#
# # Assigning gift-giving pairs
# Mails = {"Lene":"lene.idsenga@gmail.com", "Jabob": "jakob949@hotmail.com",
#          "Far": "tarup-mark@hotmail.com", "Mor": "irenekleinlarsen@hotmail.com",
#          "Martin": "martin.lunden@pwc.com", "Sara": "sara_sofie_larsen@hotmail.com",
#          "Alexander":"Alexander.faurschou@gmail.com", "Nanna": "nanna.lunden@maersk.com"}
#
# Giver_gave = ["Far", "Mor", "Lene", "Martin", "Nanna", "Alexander", "Sara", "Jabob"]
# Modtager_gave = Giver_gave.copy()
#
# for person in Giver_gave:
#     gave = random.choice(Modtager_gave)
#     # Check for specific pair conditions
#     while (
#         (person == "Far" and gave == "Mor") or
#         (person == "Mor" and gave == "Far") or
#         (person == "Martin" and gave == "Nanna") or
#         (person == "Nanna" and gave == "Martin") or
#         (person == "Alexander" and gave == "Sara") or
#         (person == "Sara" and gave == "Alexander") or
#         (person == gave)
#     ):
#         gave = random.choice(Modtager_gave)
#
#     Modtager_gave.remove(gave)
#
#     # Send email to each person
#     send_to_email = Mails[person]
#     # send_to_email = "jakob949@hotmail.com"
#     subject = f"{person}! Hvem er du julemand/julekvinde for?"
#     message = f"Du skal give en gave til {gave}. \n\nPS jeg havde lavet en stavefejl i mors mail. Så det er denne mail der er den rigtige"
#
#     # Setup the MIME
#     msg = MIMEMultipart()
#     msg['From'] = email
#     msg['To'] = send_to_email
#     msg['Subject'] = subject
#     msg.attach(MIMEText(message, 'plain'))
#
#     # Send the email
#     server.sendmail(email, send_to_email, msg.as_string())
#
# # Logout of the server
# server.quit()