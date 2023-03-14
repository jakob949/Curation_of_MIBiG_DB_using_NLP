import spacy
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

# load the English language model

def generate_wordcloud(filename):
    def lines(filename):
        # generator function to yield each line of the file as a separate string
        with open(filename, "r") as f:
            for line in f:
                yield line.replace("Title: ", "").replace("Abstract: ", "")

    # process the lines of the file using spaCy
    doc = nlp.tokenizer.pipe(lines(filename), batch_size=10000)

    # generate a frequency dictionary of words in the text
    word_freq = {}
    for tokens in doc:
        for token in tokens:
            if token.is_alpha and not token.is_stop:
                if token.text.lower() in word_freq:
                    word_freq[token.text.lower()] += 1
                else:
                    word_freq[token.text.lower()] = 1

    # generate a word cloud from the frequency dictionary
    wc = WordCloud(width=1800, height=1000, background_color="white", max_words=180, min_font_size=7)
    wc.generate_from_frequencies(word_freq)
    wc.to_file(f'{filename}.png')
    # display the word cloud
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def get_word_vectors(filename, batch_size=10000):
    # generator function to yield each line of the file as a separate string
    def lines(filename):
        with open(filename, "r") as f:
            for line in f:
                yield line

    # process each line of the file using spaCy and yield the word vectors for each sentence
    for line in lines(filename):
        doc = nlp(line)
        for sent in doc.sents:
            sent_vectors = [word.vector for word in sent if word.has_vector]
            if sent_vectors:
                sent_vectors = np.array(sent_vectors)
                sent_mean = np.mean(sent_vectors, axis=0)
                yield sent_mean

def plot_word_vectors(filename1, filename2):
    word_vectors1 = list(get_word_vectors(filename1))
    word_vectors2 = list(get_word_vectors(filename2))

    # concatenate the word vectors into a single matrix
    word_vectors = np.concatenate([word_vectors1, word_vectors2], axis=0)

    # perform PCA on the word vectors
    pca = PCA(n_components=2)
    pca_vectors = pca.fit_transform(word_vectors)

    # plot the PCA vectors
    plt.scatter(pca_vectors[:len(word_vectors1), 0], pca_vectors[:len(word_vectors1), 1], label="Positive dataset")
    plt.scatter(pca_vectors[len(word_vectors1):, 0], pca_vectors[len(word_vectors1):, 1], label="Negative dataset")
    plt.title("PCA Word Vectors - SpcaCy Embeddings")
    plt.legend()
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(f"PCA_spacy_embeddings.svg", format="svg", bbox_inches="tight")

    plt.show()

# # load the English language model
# nlp = spacy.load("en_core_web_sm")
def most_common_words(filename):
    # open the text file and process it line by line
    with open(filename, "r") as f:
        # initialize variables to keep track of the analysis
        num_sentences = 0
        num_tokens = 0
        word_freq = {}

        # iterate through each line in the file
        for line in f:
            # process the line using spaCy
            doc = nlp(line.replace("Title: ", "").replace("Abstract: ", ""))

            # update the analysis with the results for this line
            num_sentences += len(list(doc.sents))
            num_tokens += len(doc)
            for token in doc:
                if token.is_alpha and not token.is_stop:
                    if token.text.lower() in word_freq:
                        word_freq[token.text.lower()] += 1
                    else:
                        word_freq[token.text.lower()] = 1

    # print out the analysis of the entire text file
    print(f"Number of sentences: {num_sentences}")
    print(f"Number of tokens: {num_tokens}")
    print(f"Most common words:")
    sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    for word, freq in sorted_freq[:10]:
        print(f"{word}: {freq}")

def preprocess(text):
    # preprocess the text using spaCy
    doc = nlp(text)

    # extract the lemmatized tokens
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]

    return tokens

def get_topics(filename, num_topics=5, num_passes=10):
    # generator function to yield each line of the file as a separate string
    def lines(filename):
        with open(filename, "r") as f:
            for line in f:
                yield line

    # preprocess the text in each line of the file
    texts = [preprocess(line) for line in lines(filename)]

    # create a dictionary from the preprocessed texts
    dictionary = Dictionary(texts)

    # convert the preprocessed texts to bag-of-words format
    corpus = [dictionary.doc2bow(text) for text in texts]

    # perform topic modeling using the LDA algorithm
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=num_passes)

    # print the topics
    for topic in lda_model.print_topics():
        print(topic)
def analyze_topics(lda_model, corpus, num_topics, topn=10):
    # Get the topics and their top words
    topics = lda_model.show_topics(num_topics=num_topics, num_words=topn, formatted=False)

    # Create a dictionary to hold the topic information
    topic_info = {}

    # Loop over the topics and their top words
    for topic_num, topic_words in topics:
        # Get the top words for the topic
        top_words = [word for word, _ in topic_words]

        # Get the topic distribution for each document in the corpus
        topic_distribution = [dict(lda_model[d]) for d in corpus]

        # Calculate the average topic weight for the documents in the corpus
        avg_topic_weight = sum([d.get(str(topic_num), 0.0) for d in topic_distribution]) / len(topic_distribution)

        # Add the topic information to the dictionary
        topic_info[topic_num] = {
            "top_words": top_words,
            "avg_topic_weight": avg_topic_weight
        }

    # Print out the topic information
    for topic_num, info in topic_info.items():
        print(f"Topic {topic_num}:")
        print(f"  Top words: {', '.join(info['top_words'])}")
        print(f"  Avg. topic weight: {info['avg_topic_weight']}")

# load the English language model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.add_pipe("sentencizer")

# generate_wordcloud("dataset_positives_titles_abstracts.txt")
# generate_wordcloud("dataset_negatives_titles_abstracts.txt")

# plot_word_vectors('dataset_positives_titles_abstracts.txt', 'dataset_negatives_titles_abstracts.txt')

# get_topics("spacy.txt")

# most_common_words("dataset_positives_titles_abstracts.txt")
# most_common_words("dataset_negatives_titles_abstracts.txt")
