from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Read in the data from the two labeled files
with open('dataset_negatives_titles_abstracts.txt', 'r') as f:
    data_0 = f.readlines()
with open('dataset_positives_titles_abstracts.txt', 'r') as f:
    data_1 = f.readlines()

data = data_0 + data_1

# Extract the labels and text from the data
labels = np.array([int(d.split('\t')[1].strip()) for d in data])
text = np.array([d.split('\t')[0].strip() for d in data])

# Convert the text into a matrix of TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text)

# Split the data into training and testing sets
split = int(0.8 * len(data))
X_train, X_test = X[:split], X[split:]
y_train, y_test = labels[:split], labels[split:]

# Train the SVM classifier on the training data
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Print the accuracy of the classifier
print('Accuracy:', round(accuracy_score(y_test, y_pred), 2), 'F1:', round(f1_score(y_test, y_pred),2))

