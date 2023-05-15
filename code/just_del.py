# import matplotlib.pyplot as plt
#
# # List of file names
# files = ["scores_070523_flan_base.txt", "scores_080523_plain_flan_base.txt"]
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
#     for i, metric_name in enumerate(metric_names):
#         for file in data.keys():
#             metric_name_key = metric_name.replace('-', '_')
#             plt.subplot(2, 2, i+1)
#             plt.plot(data[file]["epochs_train"], data[file][f"{metric_name_key}_train"], label=f'Train {file}')
#             #plt.plot(data[file]["epochs_test"], data[file][f"{metric_name_key}_test"], label=f'Test {file}')
#             plt.xlabel("Epoch")
#             plt.ylabel(metric_name)
#             plt.title(f"{metric_name} vs. Epoch")
#             plt.legend()
#
#     plt.tight_layout()
#     plt.suptitle("Plain T5 vs ESM2-T5 - Train data", fontsize=16, y=1.05)
#     plt.savefig('Plain_T5_vs_ESM2_T5_Train_data.pdf', bbox_inches='tight', dpi=300, format='pdf')
#
#     plt.show()
#
#
# # Create a single plot with 4 subplots arranged in a 2x2 grid
# plot_metrics(data, metric_names)

with open("dataset_combined_multitask_incl_protein_seq.txt", "w") as out_file:
    with open("dataset/test_combined_multitask_incl_protein_seq.txt", "r") as test_file:
        with open("train_combined_multitask_incl_protein_seq.txt", "r") as file:
            for line in file:
                if line != '\n':
                    out_file.write(line)
            for line in test_file:
                if line != '\n':
                    out_file.write(line)


