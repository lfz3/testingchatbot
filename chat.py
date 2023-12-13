from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import torch
import random
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding="utf8") as file:
    intents = json.load(file)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
words = data["words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

probability = 0.50

#print("Welcome to my ChatBot")
#user_name = input("Type your name here and press Enter : ")
#print("Talk to our fbot and he will answer :D \nType 'exit_chat' if you want to exit")

def get_response(sentence):
    sentence = tokenize(sentence)
    bag = bag_of_words(sentence, words)
    bag = bag.reshape(1, bag.shape[0])
    bag = torch.from_numpy(bag).to(device)

    output = model(bag)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probabilities = torch.softmax(output, dim=1)
    prob = probabilities[0][predicted.item()]

    if(prob.item() >= probability):
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
    else:
        return (f"I don't understand that, sorry...")