import random
import numpy as np

list_of_words = []
list_of_words.append("<OOV>")
assigned_numbers = {}

#Convert sentences to numbers during training
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

    return tokenized

def tokenize_prediction(sentence):
    words = sentence.split(" ")
    tokenized_sentence = [assigned_numbers.get(word, 0) for word in words]
    return tokenized_sentence

def encode_to_binary(sentences):
    vocab_size = len(list_of_words)
    encoded_sentences = []
    for sentence in sentences:
        encoded_sentence = []
        for token in sentence:
            binary_representation = [0] * vocab_size
            binary_representation[token] = 1
            encoded_sentence.append(binary_representation)
        encoded_sentences.append(encoded_sentence)
    
    return encoded_sentences

def normalize_labels(labels):
    min_label = min(labels)
    max_label = max(labels)
    normalized_labels = [(label - min_label) / (max_label - min_label) for label in labels]
    return normalized_labels, min_label, max_label

def denormalize_label(label, min_label, max_label):
    return label * (max_label - min_label) + min_label

def train(sentences, labels, learning_rate=0.01, max_iterations=1000):
    sentences = tokenize(sentences)
    labels, min_label, max_label = normalize_labels(labels)
    num_features = len(sentences[0])
    
    # Initialize weights and bias
    weights = np.random.uniform(0, 1, num_features)
    bias = random.uniform(0, 1)

    for iteration in range(max_iterations):
        total_error = 0

        for sentence, label in zip(sentences, labels):
            learn = np.array(sentence)
            out = np.dot(weights, learn) + bias

            error = label - out
            total_error += error**2

            # Update weights and bias
            weights += learning_rate * error * learn
            bias += learning_rate * error

        if total_error < 1e-6:  # Convergence criterion
            break

        if iteration % 100 == 0:
            print(f'Iteration {iteration}, Total Error: {total_error}')

    return weights, bias, min_label, max_label




def predict(sentence, weights, bias, min_label, max_label):
    tokenized_sentence = tokenize_prediction(sentence)
    max_length = len(weights)
    padded_sentence = tokenized_sentence + [0] * (max_length - len(tokenized_sentence))
    result = np.dot(weights, padded_sentence) + bias
    return denormalize_label(result, min_label, max_label)




# Open the text file in read mode
with open('sentences.txt', 'r', encoding='utf-8') as file:
    # Read the lines of the file and create a list
    lines = file.readlines()

# Remove newline characters from each line
sentences = [line.strip() for line in lines]
# Open the text file in read mode
with open('labels.txt', 'r') as file:
    # Read the lines of the file and create a list
    labels = file.readlines()
# Remove newline characters from each line
labels = [int(line.strip()) for line in labels]


# Train the model
weights, bias, min_label, max_label = train(sentences, labels)


# Print the final weights and bias
print(f'Final Weights: {weights}, Final Bias: {bias}')



while true:
    new_sentence = input("Enter a sentence: ")
    prediction = predict(new_sentence, weights, bias, min_label, max_label)
    print(f'Prediction for "{new_sentence}": {prediction}')
