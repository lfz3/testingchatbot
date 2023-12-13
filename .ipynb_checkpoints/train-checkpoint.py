import json
from nltk_utils import tokenize, lemm, remove_punctuation, bag_of_words
import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json', 'r', encoding="utf8") as file:
    intents = json.load(file)

all_words = []
tags = []
pattern_with_tag = []

# Extract the words from the patterns in intents.json and lemmatize them
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokenized_words = tokenize(pattern)
        all_words.extend(tokenized_words)
        pattern_with_tag.append((tokenized_words, tag))

all_words = remove_punctuation(all_words)
all_words = [lemm(word) for word in all_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

patterns_train = []
tags_train = []

# Make bag of words array for each word
for (pattern_sentence, tag) in pattern_with_tag:
    bag = bag_of_words(pattern_sentence, all_words)
    patterns_train.append(bag)

    label = tags.index(tag)
    tags_train.append(label)

patterns_train = numpy.array(patterns_train)
tags_train = numpy.array(tags_train)

# Create custom dataset for our problem
class ChatbotDataset(Dataset):
    def __init__(self):
        self.samples_no = len(patterns_train)
        self.patterns_data = patterns_train
        self.tags_data = tags_train

    def __getitem__(self, index):
        return self.patterns_data[index], self.tags_data[index]

    def __len__(self):
        return self.samples_no

# Hyper-parameters
batch_size = 4
hidden_size = 8
input_size = len(patterns_train[0])
output_size = len(tags)
learning_rate = 0.005
num_epochs = 1000

dataset = ChatbotDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# create neuralnet model and the device on which we will perform the tasks
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Training...")
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "words": all_words,
    "tags": tags,
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training done, with file saved to {FILE}')