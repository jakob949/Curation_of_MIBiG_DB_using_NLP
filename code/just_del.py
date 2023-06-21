# import matplotlib.pyplot as plt
#
# # List of file names
# files = ["scores_160523_GT4SD_multitask-text-and-chemistry-t5-small-augm.txt", "scores_070523_molt5_base.txt"]
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

from Bio.Blast import NCBIWWW
from Bio import SeqIO


import time
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from Bio import SeqIO
from Bio.Blast import NCBIWWW

s = time.time()

def search_homologous_sequences(args):
    fasta_string, outname = args
    if int(outname.split('_')[3]) >= 850:
        database="nr"
        e_value=0.01
        fasta_string_with_header = f">seq\n{fasta_string}"
        fasta_io = StringIO(fasta_string_with_header)
        try:
            record = SeqIO.read(fasta_io, format="fasta")
            result_handle = NCBIWWW.qblast("blastp", database, record.seq, expect=e_value)
            with open(f"blastp_temp_files/{outname}.txt", "w") as out_handle:
                out_handle.write(result_handle.read())
                print(out_handle)
            result_handle.close()
        except Exception as e:
            print(f"An error occurred: {e}")
            return

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

import re
import os
import argparse
from Bio.Align import AlignInfo
from Bio import AlignIO
import math

def shannon_entropy(list_input):
    unique_base = set(list_input)
    entropy = 0
    for base in unique_base:
        p_x = list_input.count(base) / len(list_input)
        if p_x > 0:
            entropy += - p_x * math.log2(p_x)
    return entropy

def process_files(start, end, job_id):
    # Read the sorted list of files from the stored list
    with open("file_order.txt", "r") as f:
        file_list = [line.strip() for line in f]

    file_list = file_list[start:end]

    print("start")
    for ii, filename in enumerate(file_list, start=start):
        with open(f"blast_rest/{filename}", "r") as f:
            query_list = []
            subject_list = []
            q_string, s_string = "", ""
            for line in f:
                if line.startswith('>'):
                    if q_string and s_string:
                        if len(query_list) == 0:
                            query_list.append(re.sub(r'\s', '', q_string))
                        subject_list.append(re.sub(r'\s', '', s_string))
                    q_string, s_string = "", ""

                if line.startswith('Query '):
                    query = line[10:].strip()
                    query = ''.join([i for i in query if not i.isdigit()])
                    q_string += query

                elif line.startswith('Sbjct '):
                    subject = line[10:].strip()
                    subject = ''.join([i for i in subject if not i.isdigit()])
                    s_string += subject

            if q_string and s_string:
                if len(query_list) == 0:
                    query_list.append(re.sub(r'\s', '', q_string))
                subject_list.append(re.sub(r'\s', '', s_string))

            subject_list.append(query_list[0])

            with open(f"input_sequences_{job_id}.txt", "w") as f:
                for i, seq in enumerate(subject_list):
                    f.write(f">seq_{i}\n{seq}\n")
            file_path = os.getcwd()
            data_path = os.getcwd()
            file = f'input_sequences_{job_id}.txt'
            os.system(f"clustalo -i {file_path}/{file} --dealign -o {data_path}/{file[:-4]}.fasta --force --threads=10")
            alignment = AlignIO.read(f"input_sequences_{job_id}.fasta", "fasta")
            summary_align = AlignInfo.SummaryInfo(alignment)
            scores = []
            for i in range(len(alignment[0])):
                column_bases = alignment[:, i]
                scores.append(shannon_entropy(column_bases))

            consensus = summary_align.dumb_consensus()

            while len(consensus) > 850:
                index = scores.index(max(scores))
                consensus = consensus[:index] + consensus[index + 1:]
                del scores[index]

            shorten = consensus
            print(len(shorten))
            with open(f"shorten/shorten_{filename}", "w") as out:
                print(shorten , file=out, end="")




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


import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel, AdamW
import time
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse as arg
from torchmetrics.text import BLEUScore, ROUGEScore
from torchmetrics import CharErrorRate, SacreBLEUScore
from rdkit import Chem

def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

parser = arg.ArgumentParser()
parser.add_argument("-o", "--output_file_name", type=str, default="unknown", )
args = parser.parse_args()


class Dataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=851):
        self.file_path = file_path
        self.data = self.load_data()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self):
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                text = line.split(': ')[1].split('\t')[0]
                label = line.split('\t')[1].strip('\n')
                text_list = text.split('_')

                # Check if any element in text_list is longer than 2000 characters
                if all(len(element) <= 851 for element in text_list):
                    data.append((text_list, label))
                else:
                    truncated_text_list = [element[:851] for element in text_list]
                    data.append((truncated_text_list, label))

        print(len(data))
        return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        input_encoding = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding="max_length")
        target_encoding = self.tokenizer(label, return_tensors="pt", max_length=400, padding="max_length",
                                         truncation=True)

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }


start_time = time.time()

# Assume you have a T5 model and tokenizer already
T5_model_name = 'GT4SD/multitask-text-and-chemistry-t5-base-augm'
t5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)
t5_model = torch.load("model_200623_T5_v11_saving_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)


train_dataset = Dataset("dataset/invalid2validSMILE/train_invalid2validSMILE.txt", t5_tokenizer)
test_dataset = Dataset("dataset/invalid2validSMILE/test_invalid2validSMILE_ex1.txt", t5_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


learning_rate = 5e-5
optimizer = AdamW(list(t5_model.parameters()), lr=learning_rate)

rouge = ROUGEScore()
bleu = BLEUScore()
char_error_rate = CharErrorRate()
sacre_bleu = SacreBLEUScore()

num_epochs = 18
t5_model.eval()
# Training loop
for epoch in range(num_epochs):

    # Similar loop for testing

    rouge_test_accumulated = 0.0
    bleu_test_accumulated = 0.0
    char_error_rate_test_accumulated = 0.0
    sacre_bleu_test_accumulated = 0.0
    num_test_batches = 0
    Num_correct_val_mols_test = 0
    test_outputs = []

    for batch in test_loader:
        num_test_batches += 1
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = t5_model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            test_predicted_labels = t5_tokenizer.decode(outputs.logits[0].argmax(dim=-1).tolist(),
                                                        skip_special_tokens=True)
            test_true_labels = [t5_tokenizer.decode(label.tolist(), skip_special_tokens=True) for label in batch["labels"]]

            test_outputs.append({"predicted_label": test_predicted_labels, "true_label": test_true_labels[0]})

            if is_valid_smiles(test_predicted_labels):
                Num_correct_val_mols_test += 1

            with open(f"predictions_{args.output_file_name}.txt", "a") as predictions_file:
                print(f"Epoch {epoch + 1}/{num_epochs}\tTrue: {test_true_labels}\tPred: {test_predicted_labels}",
                      file=predictions_file)

            test_rouge_score = rouge(test_predicted_labels, test_true_labels)["rouge1_fmeasure"]
            test_bleu_score = bleu(test_predicted_labels.split(), [test_true_labels[0].split()])
            test_char_error_rate_score = char_error_rate(test_predicted_labels, test_true_labels).item()
            test_sacre_bleu_score = sacre_bleu([test_predicted_labels], [test_true_labels]).item()

            rouge_test_accumulated += test_rouge_score
            bleu_test_accumulated += test_bleu_score
            char_error_rate_test_accumulated += test_char_error_rate_score
            sacre_bleu_test_accumulated += test_sacre_bleu_score

    # Print and save results for this epoch
    with open(f"scores_{args.output_file_name}.txt", "a") as scores_file:

        print(
            f"Epoch {epoch + 1}/{num_epochs}\t Avg Test ROUGE-1 F1 Score\t {rouge_test_accumulated / num_test_batches}\tAvg Test BLEU Score\t {bleu_test_accumulated / num_test_batches}\tAvg Test Char Error Rate\t {char_error_rate_test_accumulated / num_test_batches}\tAvg Test SacreBLEU Score\t {sacre_bleu_test_accumulated / num_test_batches}\tNum correct val mols test: {Num_correct_val_mols_test}",
            file=scores_file)
    # save the model
    # if epoch == 17:
    #     torch.save(t5_model, f"model_{args.output_file_name}.pt")
