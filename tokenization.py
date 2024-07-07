import random
import numpy as np
from sklearn.preprocessing import StandardScaler

list_of_words = ["<OOV>"]
assigned_numbers = {}

# Convert sentences to numbers during training and assign numbers to them
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

def train(sentences, labels, learning_rate=0.00001, max_iterations=300, batch_size=32, patience=10):
    sentences = tokenize(sentences)
    labels, min_label, max_label = normalize_labels(np.array(labels))
    num_features = sentences.shape[1]

    # Standardize input
    scaler = StandardScaler()
    sentences = scaler.fit_transform(sentences)

    # Initialize weights and bias
    weights = np.random.uniform(-0.1, 0.1, num_features)
    bias = random.uniform(-0.1, 0.1)

    # Adam optimizer parameters
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8
    m_w, v_w = np.zeros_like(weights), np.zeros_like(weights)
    m_b, v_b = 0, 0
    t = 0

    best_error = float('inf')
    patience_counter = 0

    for iteration in range(max_iterations):
        total_error = 0

        # Shuffle data
        indices = np.arange(len(sentences))
        np.random.shuffle(indices)
        sentences = sentences[indices]
        labels = labels[indices]

        for start in range(0, len(sentences), batch_size):
            end = start + batch_size
            batch_sentences = sentences[start:end]
            batch_labels = labels[start:end]

            outputs = np.dot(batch_sentences, weights) + bias
            errors = batch_labels - outputs

            total_error += np.sum(errors**2)

            # Compute gradients
            gradients_w = -2 * np.dot(batch_sentences.T, errors) / batch_size
            gradients_b = -2 * np.sum(errors) / batch_size

            # Adam optimizer update
            t += 1
            m_w = beta1 * m_w + (1 - beta1) * gradients_w
            v_w = beta2 * v_w + (1 - beta2) * (gradients_w**2)
            m_w_hat = m_w / (1 - beta1**t)
            v_w_hat = v_w / (1 - beta2**t)

            m_b = beta1 * m_b + (1 - beta1) * gradients_b
            v_b = beta2 * v_b + (1 - beta2) * (gradients_b**2)
            m_b_hat = m_b / (1 - beta1**t)
            v_b_hat = v_b / (1 - beta2**t)

            weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        # Add L2 regularization term
        regularization_term = 0.01 * np.sum(weights**2)
        total_error += regularization_term

        if np.isnan(total_error):
            print("Encountered NaN values in total error. Stopping training.")
            break

        if total_error < best_error:
            best_error = total_error
            current_best = [weights, bias, min_label, max_label, scaler, best_error]

            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter > patience:
            #print(f"Early stopping at iteration {iteration}, best total error: {best_error}")
            pass
        if total_error < 1e-6:  # Convergence criterion
            break

        if iteration % 100 == 0:
            
            print(f'Iteration {iteration}, Total Error: {total_error}')
            print(current_best[5])
    return current_best[0], current_best[1], current_best[2], current_best[3], current_best[4]

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
