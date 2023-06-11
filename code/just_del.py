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

def contains_gap(filename):
    with open(filename, "r") as f:
        for line in f:
            if "-" in line:
                return True
    return False


def shannon_entropy(list_input):
    unique_base = set(list_input)
    entropy = 0
    for base in unique_base:
        p_x = list_input.count(base) / len(list_input)
        if p_x > 0:
            entropy += - p_x * math.log2(p_x)
    return entropy


import re
import os
from Bio.Align import AlignInfo
from Bio import AlignIO
import math

for ii, filename in enumerate(os.listdir("blast/")):
    file_identifier = filename.split('_')[2:]
    file_identifier[-1] = file_identifier[-1].split('.')[0]
    print(ii)
    with open(f"../blast/{filename}", "r") as f:
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

        with open("input_sequences.txt", "w") as f:
            for i, seq in enumerate(subject_list):
                f.write(f">seq_{i}\n{seq}\n")


        # Define the paths and execute the Clustal Omega command
        file_path = os.getcwd()
        data_path = os.getcwd()
        file = 'input_sequences.txt'
        os.system(f"clustalo -i {file_path}/{file} --dealign -o {data_path}/{file[:-4]}.fasta --force --threads=10")

        # Now read the alignment after the appropriate alignment has been done
        alignment = AlignIO.read("input_sequences.fasta", "fasta")
        summary_align = AlignInfo.SummaryInfo(alignment)

        scores = []
        for i in range(len(alignment[0])):
            column_bases = alignment[:, i]
            scores.append(shannon_entropy(column_bases))

        # Use the dumb_consensus method to get the consensus sequence
        consensus = summary_align.dumb_consensus()


        while len(consensus) > 850:
            index = scores.index(max(scores))
            consensus = consensus[:index] + consensus[index + 1:]
            del scores[index]

        shorten = consensus


        # First read all lines from the file and close it
        with open("Transformer_DB_Curation_MIBiG/code/dataset/protein_SMILE/dataset_protein_peptides_complete_v3_shorten.txt", "r") as infile:
            lines = infile.readlines()

        # Now open the file in write mode to write back the modified lines
        with open("Transformer_DB_Curation_MIBiG/code/dataset/protein_SMILE/dataset_protein_peptides_complete_v3_shorten.txt", "w") as outfile:
            for i, line in enumerate(lines):
                d1 = line.split('\t')[0]
                smile = line.split('\t')[1]
                data = d1.split('_')[1:]

                if i == int(file_identifier[0]):

                    # Check - To insure correct file
                    if len(data[int(file_identifier[1])]) == int(file_identifier[2]):
                        print("replaced it")
                        data[int(file_identifier[1])] = shorten
                        print(type(d1), type(data))
                        # Convert the Seq object to a string
                        data = [str(element) for element in data]
                        # Recreate the string for the line
                        new_d1 = '_'.join([d1.split('_')[0]] + data)
                        line = '\t'.join([new_d1, smile])

                outfile.write(line)

