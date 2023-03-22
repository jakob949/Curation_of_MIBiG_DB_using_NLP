import spacy
from wordcloud import WordCloud
import numpy as np
from sklearn.decomposition import PCA
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from pybliometrics.scopus import AbstractRetrieval
from textwrap import shorten
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, RobertaForSequenceClassification, RobertaTokenizer, BertTokenizer, BertForSequenceClassification, AdamW
import time
def generate_wordcloud(filename):
    """
    Given a text file, generate a word cloud image and save it to a file.
    """
    nlp = spacy.load("en_core_web_sm")

    def lines(filename):
        """
        Generator function to yield each line of the file as a separate string.
        """
        with open(filename, "r") as f:
            for line in f:
                yield line.replace("Title: ", "").replace("Abstract: ", "")

    # Process the lines of the file using spaCy
    doc = nlp.tokenizer.pipe(lines(filename), batch_size=10000)

    # Generate a frequency dictionary of words in the text
    word_freq = {}
    for tokens in doc:
        for token in tokens:
            if token.is_alpha and not token.is_stop:
                if token.text.lower() in word_freq:
                    word_freq[token.text.lower()] += 1
                else:
                    word_freq[token.text.lower()] = 1

    # Generate a word cloud from the frequency dictionary
    wc = WordCloud(width=1800, height=1000, background_color="white", max_words=180, min_font_size=7)
    wc.generate_from_frequencies(word_freq)
    wc.to_file(f'{filename}.png')
    # Display the word cloud
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def get_word_vectors(filename, batch_size=10000):
    """
    Given a text file, yield the mean word vectors for each sentence in the file.
    """
    nlp = spacy.load("en_core_web_sm")

    # Generator function to yield each line of the file as a separate string
    def lines(filename):
        with open(filename, "r") as f:
            for line in f:
                yield line

    # Process each line of the file using spaCy and yield the word vectors for each sentence
    for line in lines(filename):
        doc = nlp(line)
        for sent in doc.sents:
            sent_vectors = [word.vector for word in sent if word.has_vector]
            if sent_vectors:
                sent_vectors = np.array(sent_vectors)
                sent_mean = np.concatenate(sent_vectors, axis=0)
                yield sent_mean


def plot_word_vectors(filename1, filename2):
    """
    Given two text files, plot the word vectors using PCA.
    """
    nlp = spacy.load("en_core_web_sm")

    word_vectors1 = list(get_word_vectors(filename1))
    word_vectors2 = list(get_word_vectors(filename2))

    # Concatenate the word vectors into a single matrix
    word_vectors = np.concatenate([word_vectors1, word_vectors2], axis=0)

    # Perform PCA on the word vectors
    pca = PCA(n_components=2)
    pca_vectors = pca.fit_transform(word_vectors)

    # Plot the PCA vectors
    plt.scatter(pca_vectors[:len(word_vectors1), 0], pca_vectors[:len(word_vectors1), 1], label="Positive dataset")
    plt.scatter(pca_vectors[len(word_vectors1):, 0], pca_vectors[len(word_vectors1):, 1], label="Negative dataset")
    plt.title("PCA Word Vectors - SpcaCy Embeddings")
    plt.legend()
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(f"PCA_spacy_embeddings.svg", format="svg")

def fetch_journal(pubmed_id):

    """
    Fetch the journal for a given PubMed ID from ncbi. Very SLOW but works on all journals
    :param: pubmed_id
    :return: journal_title
    """
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    journal_section = soup.find_all("div", {"class": "journal-actions dropdown-block"})
    try:
        journal_title = journal_section[0].text.strip().split('\n')[0]
        return journal_title
    except:
        return None
def fetch_journal2(pubmed_id):
    # faster way to get the journal title
    try:
        ab = AbstractRetrieval(pubmed_id)
        title = ab.publicationName
    except:
        title = fetch_journal(pubmed_id)
        return title
    return title



def journal_distribution(filename1, filename2, minimum_num_journals=10):
    """
    Given two files of PubMed IDs, compare the distribution of journals of the articles.
    """
    def process_file(filename):
        with open(filename, 'r') as f:
            dist = {}
            for line in f:
                # function which returns the journal title
                journal = fetch_journal2(line.strip())
                if journal in dist:
                    dist[journal] += 1
                elif journal is None:
                    continue
                else:
                    dist[journal] = 1
        return dist

    dist1 = process_file(filename1)
    dist2 = process_file(filename2)

    # Combine and sort the dictionary by value
    combined_dist = {k: (dist1.get(k, 0), dist2.get(k, 0)) for k in set(dist1) | set(dist2)}
    combined_dist = {k: v for k, v in sorted(combined_dist.items(), key=lambda item: sum(item[1]), reverse=True)}

    # Remove entries with less than the minimum number of publications
    combined_dist = {k: v for k, v in combined_dist.items() if sum(v) >= minimum_num_journals}

    # Plot the distribution
    labels = [shorten(k, width=40, placeholder="...") for k in combined_dist.keys()]
    file1_counts = [v[0] for v in combined_dist.values()]
    file2_counts = [v[1] for v in combined_dist.values()]

    fig_height = max(4, len(labels) * 0.2)  # Increase figure height based on the number of labels
    fig, ax = plt.subplots(figsize=(8, fig_height))
    x = np.arange(len(labels))
    width = 0.20

    ax.barh(x - width / 2, file1_counts, width, label=f'{filename1}', align='center')
    ax.barh(x + width / 2, file2_counts, width, label=f'{filename2}', align='center')

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylabel('Journals', fontsize=10)
    ax.set_xlabel('Number of publications', fontsize=10)
    ax.set_title('Distribution of journals', fontsize=12)
    ax.legend([filename1, filename2])
    ax.set_ylim(-0.5, len(labels) - 0.5)  # Adjust ylim to start at the beginning of the y-axis
    plt.tight_layout()
    plt.savefig(f'comparison_{filename1}_{filename2}.pdf', bbox_inches='tight', dpi=300, format='pdf')
    plt.show()

    return combined_dist

def most_common_words_with_relative_frequencies(document: str, n: int) -> list:
    # Remove 'Title:' and 'Abstract:'
    document = document.replace("Title:", "").replace("Abstract:", "")

    # Divide the document into chunks
    chunk_size = 1000000
    doc_chunks = [document[i:i + chunk_size] for i in range(0, len(document), chunk_size)]

    # Count the occurrences of the lemmas
    word_counter = Counter()
    for chunk in doc_chunks:
        # Process the chunk using spaCy
        doc = nlp(chunk)

        for token in doc:
            # Ignore stopwords, punctuation, whitespace, and words shorter than 3 characters
            if not token.is_stop and not token.is_punct and not token.is_space and len(token.lemma_) > 2:
                lemma = token.lemma_
                word_counter[lemma] += 1

    # Get the n most common lemmas and their counts
    most_common_lemmas = word_counter.most_common(n)

    # Calculate the total number of counted words
    total_words = sum(word_counter.values())

    # Calculate the relative frequencies and store them in a list of tuples
    most_common_lemmas_with_relative_frequencies = [(word, count / total_words) for word, count in most_common_lemmas]

    # Return a list of tuples with the most common words and their relative frequencies
    return most_common_lemmas_with_relative_frequencies


def plot_histogram(list1, list2, label1, label2):
    # Sort lists by value in descending order
    list1.sort(key=lambda x: x[1], reverse=True)
    list2.sort(key=lambda x: x[1], reverse=True)

    data1 = [value for _, value in list1]
    data2 = [value for _, value in list2]

    labels1 = [label for label, _ in list1]
    labels2 = [label for label, _ in list2]

    fig, ax = plt.subplots()

    ax.bar(labels1, data1, label=label1, alpha=0.6)
    ax.bar(labels2, data2, label=label2, alpha=0.6)

    ax.set_ylabel('Frequencies')
    ax.set_xlabel('Words')
    ax.set_title('Frequencies of words in each dataset')
    ax.legend()

    # Rotate x-axis labels
    plt.xticks(rotation=45, size=7)

    # Adjust the bottom margin to prevent x-axis labels from being cut off
    plt.subplots_adjust(bottom=0.25)
    plt.savefig('histogram.pdf', bbox_inches='tight', dpi=300, format='pdf')
    plt.show()

def PCA_last_classification_layer(file_paths = ['dataset_positives_titles_abstracts.txt', 'dataset_negatives_titles_abstracts.txt'], model_path = 'finetuned_model_roberta_4'):

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import RobertaModel, RobertaTokenizer
    import time
    import argparse
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', type=str, help='name of the data file')
    args = parser.parse_args()

    # Define the dataset
    class Dataset(Dataset):
        def __init__(self, file_paths):
            self.data = []
            for file_path in file_paths:
                with open(file_path, "r") as f:
                    for line in f:
                        text, label = line.strip().split("\t")
                        self.data.append((text, int(label)))

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    time_start = time.time()

    # Loading pre-trained model
    model = RobertaModel.from_pretrained('finetuned_model_roberta_4')
    tokenizer = RobertaTokenizer.from_pretrained('finetuned_model_roberta_4')

    # Define the dataloader
    file_paths = ['test_fold_0.txt']
    dataset = Dataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Extract embeddings from the pre-trained model
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            texts, batch_labels = batch
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            embeddings.extend(outputs.last_hidden_state[:, 0, :].detach().numpy())
            labels.extend(batch_labels.numpy())

    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot PCA results
    plt.figure(figsize=(10, 10))
    for label in set(labels):
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label)

    plt.legend()
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA Plot")
    plt.savefig('PCA_embeddings_test_fold_0.pdf', bbox_inches='tight', dpi=300, format='pdf')

    time_end = time.time()
    print(f"Time elapsed in this session: {round(time_end - time_start, 2) / 60} minutes")

PCA_last_classification_layer(file_paths=['spacy_neg.txt'])


# plot_histogram([('gene', 0.029301070159098254), ('cluster', 0.015529662225094566), ('biosynthesis', 0.012079682183656789), ('biosynthetic', 0.010213715017075659), ('produce', 0.006633845919899637), ('acid', 0.006168146134691783), ('production', 0.006050929181952391), ('sequence', 0.005667598066237082), ('analysis', 0.00557889334524511), ('product', 0.0054933566500028515)], [('protein', 0.008742150924138542), ('cell', 0.008577324293226576), ('gene', 0.006802268268020781), ('activity', 0.004732426152914737), ('study', 0.004434470320112335), ('acid', 0.003978027342202274), ('result', 0.003965348370593661), ('bind', 0.0037085991955192513), ('high', 0.0036737320235955663), ('strain', 0.003496226421074987)], "Positive dataset", "Negative dataset")

# nlp = spacy.load("en_core_web_sm")
#
# with open("dataset_positives_titles_abstracts.txt", "r") as f:
#     positives = f.read()
# most_freq_pos = most_common_words_with_relative_frequencies(positives, 30)
#
# with open("dataset_negatives_titles_abstracts.txt", "r") as f:
#     positives = f.read()
# most_freq_neg = most_common_words_with_relative_frequencies(positives, 30)

# journal_distribution("dataset_positive_pubmedID.txt", "dataset_negtive_pubmedID.txt")


# plot_histogram([('gene', 0.029301070159098254), ('cluster', 0.015529662225094566), ('biosynthesis', 0.012079682183656789), ('biosynthetic', 0.010213715017075659), ('produce', 0.006633845919899637), ('acid', 0.006168146134691783), ('production', 0.006050929181952391), ('sequence', 0.005667598066237082), ('analysis', 0.00557889334524511), ('product', 0.0054933566500028515)], [('protein', 0.008742150924138542), ('cell', 0.008577324293226576), ('gene', 0.006802268268020781), ('activity', 0.004732426152914737), ('study', 0.004434470320112335), ('acid', 0.003978027342202274), ('result', 0.003965348370593661), ('bind', 0.0037085991955192513), ('high', 0.0036737320235955663), ('strain', 0.003496226421074987)], "Positive dataset", "Negative dataset")

# plot_word_vectors("dataset_positives_titles_abstracts.txt", "dataset_negatives_titles_abstracts.txt")
