# Predictive Curation of MiBIG Database

This project focuses on using transformer models and data science techniques to perform predictive curation of the well-established database Minimal Information about Biosynthetic Gene Clusters (MiBIG). The MiBIG Database contains information about gene clusters in prokaryotes, including literature explaining the gene cluster, protein sequences of genes in the cluster, organism, product/secondary metabolite of the gene cluster, SMILES, activity, and more. The information in the MiBIG Database will be used as both training and test data for different models to accomplish various objectives.

## Repository Contents

This repository contains a collection of Python scripts that serve two main purposes:

1. Data retrieval and processing
2. Data understanding and visualization

### Data Retrieval and Processing

- `playing_ground.py`: Retrieves data from the MiBIG Database
- `get_protein_seqs_ncbi.py`: Retrieves protein sequences from the NCBI given an accession number
- `download_pubmed_abstracts.py`: Retrieves abstracts or full texts from the literature referred to in the MiBIG Database
- `validating_MiBIG_data.py`: Validates that literature references in the MiBIG Database are correct and linked to relevant information
- `shorten_words.py`: Takes a dataset with gene names in a cluster and attempts to create meaningful abbreviations to reduce memory consumption for later models.
- `CV_file_creation.py`: Shuffles data and creates n-fold partitions of a dataset for performing cross-validation later.
### Data Understanding and Visualization

- `understand_text_dataset.py`: A large collection of functions for producing various plots and visualizations, such as histograms, PCA, Wordclouds, and others
- `journal_distribution_of_pos_data.py`: Produces a plot of journal distribution

### Models

- `bioGPT.py`: GPT-2 pre-trained on PubMed articles. Used to predict if new literature is relevant for inclusion in the MiBIG DB.
- `simple_models_for_pos_neg_classifier.py`: Three simple models, majority voting, a novel model based on word frequencies in abstract, and Support Vector Machine. All three models are used to predict if new literature is relevant for inclusion in the MiBIG DB.
- `T5.py`: T5 encoder-decoder model. Can be used to predict either SMILES based on the gene names or activity of secondary metabolite based on SMILES or whether literature should be included in MiBIG.
- `ESM2_to_T5.py`: A novel approach to predicting SMILES molecule representations from all the protein sequences in the gene cluster. It uses the ESM-2 encoder embeddings passed directly to a Flan T5 decoder (or specifically pre-trained T5 decoder).

### Fun and Learning

- `Attention_made_simply.py`: A script that explains the self-attention mechanism in transformer models with simple base Python and NumPy implementations.

## Getting Started

To get started with this project, you will need to have Python installed on your system. We recommend using a virtual environment to manage dependencies. 

1. Clone this repository to your local machine.
2. Create a virtual environment and activate it.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Run the Python scripts as needed for your specific use case.

## Contributing

We welcome contributions to this project! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch with a descriptive name.
3. Make your changes and commit them with a clear and concise commit message.
4. Create a pull request, providing a detailed explanation of your changes.

## License

This project is released under the MIT License. Please see the [LICENSE.md](LICENSE) file for more information.
