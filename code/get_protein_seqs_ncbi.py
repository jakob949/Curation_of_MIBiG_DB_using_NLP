import os
from Bio import Entrez
from Bio import SeqIO
import json



def get_protein_sequence(accession):
    """
    input: accession number
    output: list of protein sequences
    Get protein sequences from NCBI Entrez database.
    """

    # required by NCBI
    Entrez.email = "jakob949@hotmail.com"

    try:
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()

        protein_sequences = []

        for feature in record.features:
            # feature.type is eiter "gene" or "CDS
            print(feature.location)
            if feature.type == "CDS":
                protein_seq = feature.qualifiers.get("translation")
                if protein_seq:
                    protein_sequences.append(protein_seq[0])

        if protein_sequences:
            return protein_sequences
        else:
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def read_mibig_data(path):
    """
    Read the MIBiG data from the MIBiG JSON file.
    """

    with open(path) as json_file:
        mibig_data = json.load(json_file)

    return mibig_data


def get_protein_sequence_and_positions(accession, start=None, end=None):
    """
    input: accession number, start and end position of the range
    output: list of protein sequences and their positions
    Get protein sequences from NCBI Entrez database and their positions.
    """

    # required by NCBI
    Entrez.email = "jakob949@hotmail.com"

    try:
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()

        protein_sequences = []
        protein_positions = []

        for feature in record.features:
            # feature.type is either "gene" or "CDS"

            if feature.type == "CDS":
                protein_seq = feature.qualifiers.get("translation")
                position = feature.location
                # print(position.start, position.end, type(position.start), type(position.end))
                if protein_seq:
                    # Check if the CDS is within the specified range, or include all if no range is specified
                    if start is None or end is None or (start >= int(position.start) and end <= int(position.end)):
                        protein_sequences.append(protein_seq[0])
                        protein_positions.append((position.start, position.end))

        if protein_sequences:
            return protein_sequences, protein_positions
        else:
            return None, None

    except Exception as e:
        print(f"Error: {e}")
        return None, None

def get_gene_names(accession):
    """
    input: accession number
    output: list of gene names
    Get gene names from NCBI Entrez database.
    """

    # required by NCBI
    Entrez.email = "jakob949@hotmail.com"

    try:
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()

        gene_names = []

        for feature in record.features:
            # feature.type is either "gene" or "CDS"
            if feature.type == "gene":
                gene_name = feature.qualifiers.get("gene")
                if gene_name:
                    gene_names.append(gene_name[0])

        if gene_names:
            return gene_names
        else:
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def get_gene_names_and_positions(accession, start=None, end=None):
    """
    input: accession number, start and end position of the range
    output: list of gene names and their positions
    Get gene names from NCBI Entrez database and their positions.
    """

    # required by NCBI
    Entrez.email = "jakob949@hotmail.com"

    try:
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()

        gene_names = []
        gene_positions = []

        for feature in record.features:
            # feature.type is either "gene" or "CDS"
            if feature.type == "gene":
                gene_name = feature.qualifiers.get("gene")
                position = feature.location
                if gene_name:
                    # Check if the gene is within the specified range, or include all if no range is specified
                    if start is None or end is None or (start >= int(position.start) and end <= int(position.end)):
                        gene_names.append(gene_name[0])
                        gene_positions.append((position.start, position.end))

        if gene_names:
            return gene_names, gene_positions
        else:
            return None, None

    except Exception as e:
        print(f"Error: {e}")
        return None, None

def get_product_names(accession):
    """
    input: accession number
    output: list of product names
    Get product names from NCBI Entrez database.
    """

    # required by NCBI
    Entrez.email = "jakob949@hotmail.com"

    try:
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()

        product_names = []

        for feature in record.features:
            # feature.type is either "gene" or "CDS"
            if feature.type == "CDS":
                product_name = feature.qualifiers.get("product")
                if product_name:
                    product_names.append(product_name[0])

        if product_names:
            return product_names
        else:
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def get_product_names_and_positions(accession, start=None, end=None):
    """
    input: accession number, start and end position of the range
    output: list of product names and their positions
    Get product names from NCBI Entrez database and their positions.
    """

    # required by NCBI
    Entrez.email = "jakob949@hotmail.com"

    try:
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()

        product_names = []
        product_positions = []

        for feature in record.features:
            # feature.type is either "gene" or "CDS"
            if feature.type == "CDS":
                product_name = feature.qualifiers.get("product")
                position = feature.location
                if product_name:
                    # Check if the product is within the specified range, or include all if no range is specified
                    if start is None or end is None or (start >= int(position.start) and end <= int(position.end)):
                        product_names.append(product_name[0])
                        product_positions.append((position.start, position.end))

        if product_names:
            return product_names, product_positions
        else:
            return None, None

    except Exception as e:
        print(f"Error: {e}")
        return None, None



#loop thruogh all json files in the folder
for file in os.listdir('data/'):

    json_data = read_mibig_data(f'data/{file}')

    acc = json_data['cluster']['loci']['accession']
    try:
        end = json_data['cluster']['loci']['end_coord']
    except:
        end = None
    try:
        start = json_data['cluster']['loci']['start_coord']  # Corrected line
    except:
        start = None
    try:
        SMILE = json_data['cluster']['compounds'][0]['chem_struct']
    except:
        SMILE = None
    #print(file, acc, start, end, SMILE)

    # output = get_protein_sequence_and_positions(acc, start=start, end=end)[0]
    output = get_product_names_and_positions(acc, start=start, end=end)[0]


    if acc is not None and output is not None and SMILE is not None:
        print()
        with open(f"dataset/geneProduct2SMILE/dataset_geneProduct2SMILES_v1.txt", "a") as f:
            f.write(f"GeneProduct2SMILE: ")
            for item in output:
                if item == output[-1]:
                    f.write(f"{item}")
                else:
                    f.write(f"{item}_")
            f.write(f"\t{SMILE}\n")
# updated
