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
    database="nr"
    e_value=0.01
    fasta_io = StringIO(fasta_string)
    try:
        record = SeqIO.read(fasta_io, format="fasta")
        result_handle = NCBIWWW.qblast("blastp", database, record.seq, expect=e_value)
        with open(f"blastp_temp_files/{outname}.txt", "w") as out_handle:
            out_handle.write(result_handle.read())
        result_handle.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        return

fasta_strings = []
outnames = []

with open("dataset_protein_peptides_complete_v2.txt", "r") as f:
    for i, line in enumerate(f):
        line = line.split('\t')[1]
        data = line.split('_')[1:]
        for j, seq in enumerate(data):
            len_ = len(seq)
            fasta_strings.append(seq)
            outnames.append(f"seq_{i}_{j}_{len_}")

args = zip(fasta_strings, outnames)

num_workers = 32

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(search_homologous_sequences, args)

e = time.time()
print(e-s)
