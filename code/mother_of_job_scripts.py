import os

#
# fasta_strings = []
# outnames = []
#
# with open("dataset/protein_SMILE/dataset_protein_peptides_complete_v3.txt", "r") as f:
#     for i, line in enumerate(f):
#         line = line.split('\t')[0]
#         data = line.split('_')[1:]
#
#         for j, seq in enumerate(data):
#             len_ = len(seq)
#             if len_ >= 850:
#                 fasta_strings.append(seq)
#
#                 outnames.append(f"seq_{str(i).zfill(4)}_{str(j).zfill(5)}_{len_}")
#
#
# args = zip(fasta_strings, outnames)
#
# # make a loop which creates a file for each item in args
# for item in args:
#     fasta_string, outname = item
#     with open(f"../../raw/raw_{outname}.fasta", "w") as f:
#         f.write(f">{outname}\n{fasta_string}")

import os
import sys

# Get the list of all files in the "raw" directory
file_list = [f for f in os.listdir("raw") if f.endswith('.fasta')]

# Sort the file list to ensure consistency across different jobs
file_list.sort()

# Get start and end indices from command-line arguments
start = int(sys.argv[1])
end = int(sys.argv[2])

# Extract identifiers and apply the blastp command
for file in file_list[start:end]:
    # Remove extension to get the identifier
    identifier = file.split(".")[0].replace("raw_", "")

    # Run the blastp command
    command = f"blastp -task blastp-fast -query raw_{identifier}.fasta -db nr -num_threads 200 -max_target_seqs 8 -out blast/result_{identifier}.txt"
    os.system(command)


