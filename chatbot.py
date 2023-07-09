import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tkinter as tk
import cv2
from PIL import Image, ImageTk

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)


class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


input_size = len(training[0])
hidden_size = 8
output_size = len(output[0])

model = ChatbotModel(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train_model(model, optimizer, criterion, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        inputs = torch.tensor(training, dtype=torch.float32)
        targets = torch.tensor(np.argmax(output, axis=1), dtype=torch.long)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


train_model(model, optimizer, criterion, epochs=1000)

torch.save(model.state_dict(), "model.pt")


class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Maggie")

        self.create_widgets()

    def create_widgets(self):
        self.video_canvas = tk.Canvas(self.root, width=640, height=480)
        self.video_canvas.pack()

        self.chat_box = tk.Text(self.root, width=50, height=10)
        self.chat_box.pack()

        self.input_entry = tk.Entry(self.root, width=50)
        self.input_entry.pack()

        self.send_button = tk.Button(
            self.root, text="Send", command=self.process_input)
        self.send_button.pack()

    def process_input(self):
        user_input = self.input_entry.get()
        self.display_message("You: " + user_input)
        response = self.generate_response(user_input)
        self.display_message("Maggie: " + response)
        self.input_entry.delete(0, tk.END)

    def generate_response(self, user_input):
        inp = user_input
        input_data = torch.tensor(bag_of_words(
            inp, words), dtype=torch.float32).unsqueeze(0)
        output = model(input_data)
        _, predicted = torch.max(output, dim=1)
        tag = labels[predicted.item()]

        for intent in data["intents"]:
            if intent['tag'] == tag:
                responses = intent['responses']
        response = random.choice(responses)

        return response

    def display_message(self, message):
        self.chat_box.insert(tk.END, message + "\n")
        self.chat_box.see(tk.END)

    def play_video(self):
        video = cv2.VideoCapture('giphy.mp4')

        while True:
            ret, frame = video.read()

            if not ret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            self.video_canvas.create_image(0, 0, image=photo, anchor=tk.NW)

            self.root.update()

        video.release()


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


root = tk.Tk()
chatbot_gui = ChatbotGUI(root)
root.after(100, chatbot_gui.play_video)
root.mainloop()
