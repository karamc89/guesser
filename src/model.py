'''
RNN concatenates all tokenized and embedded word vectors into a single higher level embedding
to later pass through a fully connected network

Fully connected network also defined in the modelRNN class

Training function defined as well
'''
import torch
import torch.nn as nn

class modelRNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_class):
        #input_size = size of input word2vec embeddings, each vector represents one word
        #hidden_size = size of hidden state vectors
        #num_class = number of possible job categories to classify to
        super(modelRNN, self).__init__()
        self.emb = nn.Embedding.from_pretrained(word2vec.vectors) #
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_class)

        def forward(self, x):
            x = self.emb(x)
            h0 = torch.zeros(1,x.size(0), self.hidden_size) #initializes the leading state vector
            out,__ = self.rnn(x, h0) 
            return self.fc(out[:,-1,:])


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

#implementation:
#train_set = get_train_set() 
#val_set = get_val_set()

model_rnn = modelRNN()

#train(model_rnn, train_set, val_set, 5, 1e-5)

print(embedded_resume)




