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
from data import df_trainval, eval_data_set
from data import word2vec

#focus on important parts of input
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, encoder_outputs):
        # encoder_outputs: (batch_size, seq_len, hidden_size*2)
        energy = self.attention(encoder_outputs)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(energy, dim=1)  # (batch_size, seq_len, 1)
        
        #Apply attention weights to encoder outputs
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)  # (batch_size, hidden_size*2)
        return context_vector, attention_weights

#RNN model 
class modelRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers=2, dropout_rate=0.3):
        #input_size = size of input word2vec embeddings, each vector represents one word
        #hidden_size = size of hidden state vectors
        #num_class = number of possible job categories to classify to
        super(modelRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,batch_first=True, bidirectional=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #attention
        self.attention = Attention(hidden_size)
        #dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        #fully connected layers with batch norm
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        #initialize
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        #LSTM output
        outputs, (_, _) = self.rnn(x, (h0, c0))
        #apply attention
        context_vector, _ = self.attention(outputs)
        #apply dropout 
        x = self.dropout(context_vector)
        #pass through fully connected layers
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


#training
def train(model, train_loader, val_loader, n_epochs, lr):
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    
    best_val_acc = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for resume, label in train_loader: 
            optimizer.zero_grad()
            prediction = model(resume)
            loss = criterion(prediction, label)
            loss.backward()
            
            #clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            predicted_labels = torch.argmax(prediction, dim=1)
            correct_train += (predicted_labels == label).sum().item()
            total_train += label.size(0)

        model.eval()
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
                total_val += label.size(0)

        #get loss
        avg_train_loss = train_loss / len(train_loader)
        #if val_loader is empty handle division by zero
        avg_valid_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        train_accuracy = correct_train / total_train if total_train > 0 else 0
        valid_accuracy = correct_val / total_val if total_val > 0 else 0

        #update learning rate
        scheduler.step(avg_valid_loss)
        
        #save best model
        if valid_accuracy > best_val_acc:
            best_val_acc = valid_accuracy
            best_model = model.state_dict().copy()

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
hidden_size = 256 
batch_size = 32
num_class = 24
num_epochs = 40  
learning_rate = 3e-4  
max_seq_length = 150 

#model
model_rnn = modelRNN(
    input_size=input_size, 
    hidden_size=hidden_size, 
    num_class=num_class,
    num_layers=2,  
    dropout_rate=0.3
)

# This makes training faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    train_set, val_set = get_data_sets(df_trainval, word2vec, max_len=max_seq_length, val_split=0.11)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    model_rnn = model_rnn.to(device)
    train(model_rnn, train_loader, val_loader, num_epochs, learning_rate)
    
    eval_set, _ = get_data_sets(eval_data_set, word2vec, max_len=max_seq_length, val_split=1.0)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    
    model_rnn.eval()
    criterion = nn.CrossEntropyLoss()
    
    eval_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for resumes, labels in eval_loader:
            resumes = resumes.to(device)
            labels = labels.to(device)
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





except Exception as e:
    import traceback
    print(f"Error occurred: {e}")
    traceback.print_exc()