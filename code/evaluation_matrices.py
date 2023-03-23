import matplotlib.pyplot as plt


def plot_predictions(predictions_model1, predictions_model2):
    assert len(predictions_model1) == len(predictions_model2), "Both lists should have the same length."

    # Prepare data for plotting
    x = list(range(len(predictions_model1)))

    # Plot predictions
    plt.scatter(x, predictions_model1, c='blue', marker='o', label='Transfomer')
    plt.scatter(x, predictions_model2, c='red', marker='x', label='SVM')

    # Customize plot appearance
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Label')
    plt.title('Comparison of Predictions Transfomer vs SVM')
    plt.legend()

    # Show plot
    plt.show()


def calculate_agreement_rate(predictions_model1, predictions_model2):
    assert len(predictions_model1) == len(predictions_model2), "Both lists should have the same length."

    agreement_count = 0
    total_count = len(predictions_model1)

    for pred1, pred2 in zip(predictions_model1, predictions_model2):
        if pred1 == pred2:
            agreement_count += 1

    agreement_rate = agreement_count / total_count
    return agreement_rate



SVM = []
with open('hard_svm_y_pred.csv', 'r') as f:
    for line in f:

        SVM.append(int(line.strip()))
import pickle
with open('hard_preds.pickle', 'rb') as handle:
    trans = pickle.load(handle)
# plot_predictions(trans, SVM)
print(calculate_agreement_rate(trans, SVM))
import statistics as math
print(math.mean(SVM), math.mean(trans))