import random
import numpy as np
from sklearn.preprocessing import StandardScaler

list_of_words = ["<OOV>"]
assigned_numbers = {}

# Convert sentences to numbers during training
def tokenize(sentences):
    global list_of_words, assigned_numbers
    for sentence in sentences:
        words = sentence.split(" ")
        for word in words:
            if word not in assigned_numbers:
                list_of_words.append(word)
                assigned_numbers[word] = len(list_of_words) - 1

    tokenized = []
    for sentence in sentences:
        words = sentence.split(" ")
        tokenized_sentence = [assigned_numbers.get(word, 0) for word in words]
        tokenized.append(tokenized_sentence)
    max_length = len(max(tokenized, key=len))
    tokenized = [array + [0] * (max_length - len(array)) for array in tokenized]

    return np.array(tokenized)

def tokenize_prediction(sentence):
    words = sentence.split(" ")
    tokenized_sentence = [assigned_numbers.get(word, 0) for word in words]
    return tokenized_sentence

def normalize_labels(labels):
    min_label = min(labels)
    max_label = max(labels)
    normalized_labels = (labels - min_label) / (max_label - min_label)
    return normalized_labels, min_label, max_label

def denormalize_label(label, min_label, max_label):
    return label * (max_label - min_label) + min_label

def train(sentences, labels, learning_rate=0.0001, max_iterations=1000):
    sentences = tokenize(sentences)
    labels, min_label, max_label = normalize_labels(np.array(labels))
    num_features = sentences.shape[1]

    # Standardize input
    scaler = StandardScaler()
    sentences = scaler.fit_transform(sentences)

    # Initialize weights and bias
    weights = np.random.uniform(0, 1, num_features)
    bias = random.uniform(0, 1)

    for iteration in range(max_iterations):
        total_error = 0
        outputs = np.dot(sentences, weights) + bias
        errors = labels - outputs

        total_error = np.sum(errors**2)

        # Update weights and bias with gradient clipping
        gradients = np.dot(errors, sentences)
        gradients = np.clip(gradients, -1, 1)
        weights += learning_rate * gradients
        bias += learning_rate * np.sum(errors)

        if np.isnan(total_error):
            print("Encountered NaN values in total error. Stopping training.")
            break

        if total_error < 1e-6:  # Convergence criterion
            break

        if iteration % 100 == 0:
            print(f'Iteration {iteration}, Total Error: {total_error}')

    return weights, bias, min_label, max_label, scaler

def predict(sentence, weights, bias, min_label, max_label, scaler):
    tokenized_sentence = tokenize_prediction(sentence)
    max_length = len(weights)
    padded_sentence = tokenized_sentence + [0] * (max_length - len(tokenized_sentence))
    padded_sentence = np.array(padded_sentence).reshape(1, -1)
    # Standardize input
    padded_sentence = scaler.transform(padded_sentence)
    result = np.dot(weights, padded_sentence.T) + bias
    return denormalize_label(result, min_label, max_label)

# Open the text file in read mode
with open('sentences.txt', 'r', encoding='utf-8') as file:
    sentences = [line.strip() for line in file.readlines()]

# Open the text file in read mode
with open('labels.txt', 'r') as file:
    labels = [int(line.strip()) for line in file.readlines()]

# Train the model
weights, bias, min_label, max_label, scaler = train(sentences, labels)

# Print the final weights and bias
print(f'Final Weights: {weights}, Final Bias: {bias}')

while True:
    new_sentence = input("Enter a sentence: ")
    prediction = predict(new_sentence, weights, bias, min_label, max_label, scaler)
    print(f'Prediction for "{new_sentence}": {prediction}')
