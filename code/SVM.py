from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

# Read in the data from the two labeled files
for k in range(5):
    train_file = f'train_fold_{k}.txt'
    test_file = f'test_fold_{k}.txt'
    with open(train_file, 'r') as f:
        train_data = f.readlines()
    with open(test_file, 'r') as f:
        test_data = f.readlines()

    # Extract the labels and text from the data
    train_labels = np.array([int(d.split('\t')[1].strip()) for d in train_data])
    train_text = np.array([d.split('\t')[0].strip() for d in train_data])
    test_labels = np.array([int(d.split('\t')[1].strip()) for d in test_data])
    test_text = np.array([d.split('\t')[0].strip() for d in test_data])

    # Convert the text into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    # Train the SVM classifier on the training data
    clf = SVC(kernel='linear')
    clf.fit(X_train, train_labels)

    # Make predictions on the testing data
    y_pred = clf.predict(X_test)

    # Compute the accuracy and F1 score of the classifier
    acc_score = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)

    # Print the accuracy and F1 score of the classifier
    print('Accuracy:', round(acc_score, 2), 'F1 score:', round(f1,2))
    print(confusion_matrix(test_labels, y_pred))