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

### Data Understanding and Visualization

- `understand_text_dataset.py`: A large collection of functions for producing various plots and visualizations, such as histograms, PCA, and others

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

This project is released under the MIT License. Please see the [LICENSE](LICENSE) file for more information.
