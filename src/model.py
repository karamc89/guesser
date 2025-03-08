'''
RNN concatenates all tokenized and embedded word vectors into a single higher level embedding
to later pass through a fully connected network

Fully connected network also defined in the modelRNN class

Training function defined as well
'''
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import random 


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
def train(model, train_set, val_set, n_epochs, lr):
    criterion = nn.CrossEntropyLoss() # can change loss criteria, but CE is most used for classification tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #can change optimizer and lr for tuning

    for epoch in range(n_epochs):
        for resume, label in train_set: #training data will contain resume content paired with predefined job category label
            optimizer.zero_grad()
            prediction = model(resume)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        correct = 0
        total = 0
        for resume, label in val_set:
            prediction = model(resume)
            val_loss += criterion(prediction, label).item()
            _, predicted = torch.max(prediction, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        avg_val_loss = val_loss / len(val_set)
        accuracy = 100 * correct / total
        print("epoch #:", epoch + 1)
        print("Loss:", avg_val_loss)
        print("Accuracy:", accuracy)
           

#function: get data sets by implementing embed function on df
def get_data_sets(df, word2vec, max_len, val_split=0.2):
    # Split into training and validation sets
    val_size = int(len(df) * val_split)
    indices = list(range(len(df)))
    random.shuffle(indices)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_data = df.iloc[train_indices]
    val_data = df.iloc[val_indices]

    train_resumes = [embed(x, word2vec, max_len) for x in train_data['Cleaned_Resume']]
    val_resumes = [embed(x, word2vec, max_len) for x in val_data['Cleaned_Resume']]

    train_set = torch.utils.data.TensorDataset(torch.tensor(train_resumes, dtype=torch.float32), torch.tensor(train_data['Job_Category'].values))
    val_set = torch.utils.data.TensorDataset(torch.tensor(val_resumes, dtype=torch.float32), torch.tensor(val_data['Job_Category'].values))

    return train_set, val_set


#implementation:

input_size = 300
hidden_size = 128
num_class = 24
model_rnn = modelRNN(input_size, hidden_size, num_class)

train_set, val_set = get_data_sets(df, word2vec, max_len = 100)
train(model_rnn, train_set, val_set, 5, 1e-5)






