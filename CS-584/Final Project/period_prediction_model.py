from tqdm.auto import tqdm
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json

MODEL_OUTPUT_PATH = './models/punctuation_model_v02.pt'
np.random.seed(42)

# LABEL 0: non-sentence
# LABEL 1: sentence

class PunctuationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, num_datapoints=None, ratio_of_sentence_to_non_sentence=0.5, minimum_sentence_length=5,):
        
        self.df = df
        self.ratio_of_sentence_to_non_sentence = ratio_of_sentence_to_non_sentence

        # mask for long sentences
        mask = df['vector'].apply(lambda x: sum(x>0) >= minimum_sentence_length)
        self.masked_df = df[mask]

        if num_datapoints is not None:
            self.num_datapoints = num_datapoints
        else:
            # use whole dataset
            self.num_datapoints = len(df)

        self.X, self.Y = self.generate_data()
    
    def generate_data(self):

        X, Y = [], []

        for i, row in self.masked_df.iterrows():

            # sentence
            if np.random.rand() < self.ratio_of_sentence_to_non_sentence:
                
                sentence = row['vector']

                X.append(sentence)
                Y.append(1)
            
            # non-sentence
            else:
                
                original_sentence = row['sentence']
                original_sentence_length = len(original_sentence.split(' '))

                sentence = row['vector']
                cut_off = np.random.randint(0, original_sentence_length-1)
                sentence[cut_off:] = 0

                X.append(sentence)
                Y.append(0)
        
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.Y[idx], dtype=torch.long)

def create_dataloaders(df, num_datapoints=None, ratio_of_sentence_to_non_sentence=0.5, minimum_sentence_length=5, batch_size=32, shuffle=True):
    
    dataset = PunctuationDataset(df, num_datapoints, ratio_of_sentence_to_non_sentence, minimum_sentence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

class LSTMPunctuationTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags, pad_idx=0):
        super(LSTMPunctuationTagger, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        
    def forward(self, x):
        # x: [batch, seq_len]
        embeds = self.embedding(x)  # [batch, seq_len, embed_dim]
        outputs, (h_n, c_n) = self.lstm(embeds)  
        # h_n: [num_layers * 2, batch, hidden_dim]
        # For a single-layer BiLSTM, h_n[0] is the last hidden state of the forward LSTM
        # and h_n[1] is the last hidden state of the backward LSTM.
        
        # Concatenate the final hidden states from both directions
        h_final = torch.cat((h_n[0], h_n[1]), dim=-1)  # [batch, hidden_dim * 2]
        
        # Apply the fully connected layer to get predictions
        logits = self.fc(h_final)  # [batch, num_tags]
        
        return logits
    
def train(train_loader, num_epochs=15, **model_kwargs):

    model = LSTMPunctuationTagger(**model_kwargs)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):

        total_loss = 0
        accuracy = 0
        count = 0

        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}', unit='batch', total=len(train_loader), leave=False):
            optimizer.zero_grad()
            logits = model(batch_x)
            pred = logits.argmax(dim=-1)

            # Flatten for loss calculation
            # logits: [batch, seq_len, num_tags]
            # targets: [batch, seq_len]
            loss = criterion(logits, batch_y)
            
            num_correct = sum(pred == batch_y)
            accuracy += int(num_correct)
            count += len(batch_y)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        accuracy /= count
        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Acc: {accuracy:.4f}")
    
    return model

def test(test_loader, model):

    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    accuracy = 0
    count = 0

    for batch_x, batch_y in tqdm(test_loader, desc=f'Evaluating', unit='batch', total=len(test_loader), leave=False):
        
        logits = model(batch_x)
        pred = logits.argmax(dim=-1)

        # Flatten for loss calculation
        # logits: [batch, seq_len, num_tags]
        # targets: [batch, seq_len]
        loss = criterion(logits, batch_y)     
        
        num_correct = sum(pred == batch_y)
        accuracy += int(num_correct)
        count += len(batch_y)

        total_loss += loss.item()
    
    accuracy /= count
    avg_loss = total_loss/len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Acc: {accuracy:.4f}")



def main():
        
        train_data = pd.read_pickle('./data/train.pkl')
        test_data = pd.read_pickle('./data/test.pkl')
    
        with open('./data/vocab_info.json', 'rb') as f:
            vocab_info = json.load(f)
        vocab = vocab_info['vocab']
    
        train_loader = create_dataloaders(
            train_data,  
            num_datapoints=None,
            ratio_of_sentence_to_non_sentence=0.5,
            minimum_sentence_length=5, 
            batch_size=128, 
            shuffle=True
        )
        test_loader = create_dataloaders(
            test_data, 
            num_datapoints=None, 
            ratio_of_sentence_to_non_sentence=0.6, 
            minimum_sentence_length=5,
            batch_size=128, 
            shuffle=True
        )
    
        model = train(train_loader, vocab_size=len(vocab), embed_dim=100, hidden_dim=64, num_tags=2)
        torch.save(model.state_dict(), MODEL_OUTPUT_PATH)

        test(test_loader, model)

if __name__ == '__main__':
    main()