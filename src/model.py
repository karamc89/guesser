'''
RNN concatenates all tokenized and embedded word vectors into a single higher level embedding
to later pass through a fully connected network

Fully connected network also defined in the modelRNN class

Training function defined as well
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import random 
import data
from data import df_trainval, eval_data_set
from data import word2vec
from data import embed

#class: RNN model 
class modelRNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_class):
        #input_size = size of input word2vec embeddings, each vector represents one word
        #hidden_size = size of hidden state vectors
        #num_class = number of possible job categories to classify to
        super(modelRNN, self).__init__()
        #self.emb = nn.Embedding.from_pretrained(word2vec.vectors) #
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        #x = self.emb(x)
        h0 = torch.zeros(1,x.size(0), self.hidden_size) #initializes the leading state vector
        out,__ = self.rnn(x, h0) 
        return self.fc(out[:,-1,:])


#function: training the model
def train(model, train_loader, val_loader, n_epochs, lr):
    criterion = nn.CrossEntropyLoss() # can change loss criteria, but CE is most used for classification tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #can change optimizer and lr for tuning

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for resume, label in train_loader: #training data will contain resume content paired with predefined job category label
            optimizer.zero_grad()
            prediction = model(resume)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            predicted_labels = torch.argmax(prediction, dim=1)
            correct_train += (predicted_labels == label).sum().item()
            total_train += label.size(0)

        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for resume, label in val_loader:
                prediction = model(resume)
                loss = criterion(prediction, label)
                val_loss += loss.item()
                predicted_labels = torch.argmax(prediction, dim=1)
                correct_val += (predicted_labels == label).sum().item()
        for resume, label in val_loader:
            prediction = model(resume)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
            val_loss += loss.item()
            predicted_labels = torch.argmax(prediction, dim=1)
            correct_val += (predicted_labels == label).sum().item()
            total_val += label.size(0)


        #get loss
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = val_loss / len(val_loader)
        train_accuracy = correct_train / total_train
        valid_accuracy = correct_val / total_val

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        train_accs.append(train_accuracy)
        valid_accs.append(valid_accuracy)


        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"  Val   Loss: {avg_valid_loss:.4f} | Val   Accuracy: {valid_accuracy:.4f}")

    plt.plot(range(1, n_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, n_epochs+1), valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    #plot accuracy
    plt.title("Train vs Validation Accuracy")
    plt.plot(range(1,n_epochs+1), train_accs, label="Train")
    plt.plot(range(1,n_epochs+1), valid_accs, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
           

#function: get data sets by implementing embed function on df
def get_data_sets(df, word2vec, max_len, val_split):
    # Split into training and validation sets
    val_size = int(len(df) * val_split)
    indices = list(range(len(df)))
    random.shuffle(indices)
    train_indices = indices[:val_size] 
    val_indices = indices[val_size:] 

    train_data = df.iloc[train_indices]
    val_data = df.iloc[val_indices]


    train_resumes = np.array([np.array(x, dtype=np.float32) for x in train_data['Embedded_Resume']], dtype=np.float32)
    val_resumes = np.array([np.array(x, dtype=np.float32) for x in val_data['Embedded_Resume']], dtype=np.float32)

    train_set = torch.utils.data.TensorDataset(torch.tensor(train_resumes, dtype=torch.float32), torch.tensor(train_data['Category'].values, dtype=torch.long))
    val_set = torch.utils.data.TensorDataset(torch.tensor(val_resumes, dtype=torch.float32), torch.tensor(val_data['Category'].values, dtype=torch.long))


    return train_set, val_set


#implementation:

input_size = 300
hidden_size = 128
batch_size = 32
num_class = 24
num_epochs = 30
model_rnn = modelRNN(input_size, hidden_size, num_class)

train_set, val_set = get_data_sets(df_trainval, word2vec, max_len = 100, val_split=0.11)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


train(model_rnn, train_loader, val_loader, num_epochs, 1e-5)

# testing/evaluation step
eval_set, _ = get_data_sets(eval_data_set, word2vec, max_len=100, val_split=1.0)
eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

model_rnn.eval()
criterion = nn.CrossEntropyLoss()

eval_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for resumes, labels in eval_loader:
        predictions = model_rnn(resumes)
        loss = criterion(predictions, labels).item()
        eval_loss += loss

        _, predicted = torch.max(predictions, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_eval_loss = eval_loss / len(eval_set)
accuracy = correct / total

print("Final Evaluation on eval_data_set:")
print(f"Loss: {avg_eval_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")