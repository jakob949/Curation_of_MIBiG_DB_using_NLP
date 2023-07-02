import os

def shorten(word):
    """Create abbreviation from a word."""
    return word[:3] if len(word) >= 3 else word

def process_file(input_file, output_file):
    """Read dataset and create abbreviations for geneProducts strings."""
    word_dict = {}
    abbreviation_dict = {}
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.split("\t")
            gene_product = parts[0].split(": ")[1]
            words = gene_product.split("_")

            abbreviations = []
            for word in words:
                # If the word has been seen before, use its abbreviation
                if word in word_dict:
                    abbreviations.append(word_dict[word])
                else:
                    abbreviation = shorten(word)

                    # Create a new unique abbreviation if needed
                    while abbreviation in abbreviation_dict:
                        abbreviation_dict[abbreviation] += 1
                        abbreviation += str(abbreviation_dict[abbreviation])
                    word_dict[word] = abbreviation
                    abbreviation_dict[abbreviation] = 0

                    abbreviations.append(abbreviation)
            abbreviated_product = "_".join(abbreviations)

            parts[0] = f'geneProduct: {abbreviated_product}'
            f_out.write('\t'.join(parts))


process_file('dataset/geneProduct2SMILE/dataset_geneProduct2SMILES_v1.txt', 'dataset/geneProduct2SMILE/dataset_geneProduct2SMILES_v1_short.txt')
