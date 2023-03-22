import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def self_attention(query, key, value, mask=None):
    # Compute attention scores (query-key dot product)
    scores = np.dot(query, key.T)

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Compute softmax of attention scores
    attention_weights = softmax(scores)

    # Compute the weighted sum of values
    output = np.dot(attention_weights, value)

    return output, attention_weights

# Example usage
# Note: In practice, Q, K, V are typically multi-dimensional arrays
query = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
key = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
value = np.array([[0, 2, 0], [0, 3, 0], [0, 5, 0]])

output, attention_weights = self_attention(query, key, value)
print("Output:", output)
print("Attention Weights:", attention_weights)
