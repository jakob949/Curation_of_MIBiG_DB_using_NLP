import spacy
from wordcloud import WordCloud
import numpy as np
from sklearn.decomposition import PCA
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from pybliometrics.scopus import AbstractRetrieval


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
                sent_mean = np.mean(sent_vectors, axis=0)
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


def journal_distribution(filename, minimum_num_journals=5):
    """
    Given a text of pubmed id, return the distribution of journals of the articles
    """
    with open(filename, 'r') as f:
        dist = {}
        for line in f:
            journal = fetch_journal2(line.strip())
            if journal in dist:
                dist[journal] += 1
            elif journal is None:
                continue
            else:
                dist[journal] = 1

        # sort the dictionary by value
        dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1], reverse=True)}

        # remove entries with less than 5 publications
        dist = {k: v for k, v in dist.items() if v >= minimum_num_journals}

        # plot the distribution
        plt.bar(range(len(dist)), list(dist.values()), align='center')
        plt.xticks(range(0, len(dist)), list(dist.keys()), rotation=90, fontsize=4)
        plt.xlabel('Journals', fontsize=10)
        plt.ylabel('Number of publications', fontsize=10)
        plt.title('Distribution of journals', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{filename}.svg', bbox_inches='tight', dpi=300, format='svg')
        plt.show()

    return dist

dist = journal_distribution("pubMedIDs_NONBGC.txt")