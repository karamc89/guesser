'''
RNN concatenates all tokenized and embedded word vectors into a single higher level embedding
to later pass through a fully connected network

Fully connected network also defined in the modelRNN class

Training function defined as well
'''
import torch
import numpy as np
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

    for epoch in range(n_epochs):
        for resume, label in train_loader: #training data will contain resume content paired with predefined job category label
            optimizer.zero_grad()
            prediction = model(resume)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        correct = 0
        total = 0
        for resume, label in val_loader:
            prediction = model(resume)
            val_loss += criterion(prediction, label).item()
            _, predicted = torch.max(prediction, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        avg_val_loss = val_loss / len(val_set)
        accuracy = correct / total
        print("epoch #:", epoch + 1)
        print("Loss:", avg_val_loss)
        print("Accuracy:", accuracy)
           

#function: get data sets by implementing embed function on df
def get_data_sets(df, word2vec, max_len, val_split):
    # Split into training and validation sets
    val_size = int(len(df) * val_split)
    indices = list(range(len(df)))
    random.shuffle(indices)
    val_indices = indices[:val_size] 
    train_indices = indices[val_size:] 

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
model_rnn = modelRNN(input_size, hidden_size, num_class)

train_set, val_set = get_data_sets(df_trainval, word2vec, max_len = 100, val_split=0.2)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


train(model_rnn, train_loader, val_loader, 5, 1e-5)

# testing/evaluation step
eval_set, _ = eval_set, _ = get_data_sets(eval_data_set, word2vec, max_len=100, val_split=0)
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

avg_eval_loss = eval_loss / len(eval_loader)
accuracy = correct / total

print("Final Evaluation on eval_data_set:")
print(f"Loss: {avg_eval_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")




