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
