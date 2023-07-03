import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TransfoXLTokenizer, TransfoXLForSequenceClassification

#BERT transformer
# Step 1: Data Collection
data = pd.read_csv('Automobile_data.csv') 

# Step 2: Data Exploration
print("Data shape:", data.shape)
print("First 5 rows of data:")
print(data.head())

# Step 3: Data Preparation
selected_columns = ['name', 'mpg', 'cylinders', 'displacement','horsepower', 'weight', 'accelaration', 'model_year', 'origin']
data = data[selected_columns]
# Dropping rows with missing values
data = data.dropna()


# Step 4: Split Data into Training and Test Sets
X = data['text']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  

# Tokenize input text
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

# Convert labels to tensors
train_labels = torch.tensor(list(y_train))
test_labels = torch.tensor(list(y_test))

# Create data loaders
train_dataset = TensorDataset(train_encodings.input_ids, train_encodings.attention_mask, train_labels)
test_dataset = TensorDataset(test_encodings.input_ids, test_encodings.attention_mask, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Set device to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
model.train()
for epoch in range(5):  # Change the number of epochs as needed
    total_loss = 0
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{5} - Average Loss: {avg_loss:.4f}")



# Step 6: Model Evaluation
model.eval()
total_correct = 0
total_samples = 0
for batch in test_loader:
    batch = tuple(t.to(device) for t in batch)
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    predicted_labels = torch.argmax(logits, dim=1)
    total_correct += (predicted_labels == inputs['labels']).sum().item()
    total_samples += len(inputs['labels'])

accuracy = total_correct / total_samples
print("Model Accuracy:", accuracy)


#GPT transformer
# Step 1: Data Collection
data = pd.read_csv('Automobile_data.csv')  

# Step 2: Data Exploration
print("Data shape:", data.shape)
print("First 5 rows of data:")
print(data.head())

# Step 3: Data Preparation

selected_columns = ['name', 'mpg', 'cylinders', 'displacement','horsepower', 'weight', 'accelaration', 'model_year', 'origin']  
data = data[selected_columns]

# Dropping rows with missing values
data = data.dropna()



# Step 4: Split Data into Training and Test Sets
X = data['text']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize input text
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

# Convert labels to tensors
train_labels = torch.tensor(list(y_train))
test_labels = torch.tensor(list(y_test))

# Create custom dataset
class CustomDataset('Automobile_data.csv'):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Set device to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
model.train()
for epoch in range(5):  # Change the number of epochs as needed
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch, labels=batch['input_ids'])
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{5} - Average Loss: {avg_loss:.4f}")


# Step 6: Model Evaluation
model.eval()
total_correct = 0
total_samples = 0
for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch, labels=batch['input_ids'])
        predicted_labels = torch.argmax(outputs.logits, dim=2)
        total_correct += (predicted_labels == batch['input_ids']).sum().item()
        total_samples += batch['input_ids'].numel()

accuracy = total_correct / total_samples
print("Model Accuracy:", accuracy)


#Transformer XL

# Step 1: Data Collection
data = pd.read_csv('Automobile_data.csv')  # Replace 'Automobile_data.csv' with your dataset file path

# Step 2: Data Exploration
print("Data shape:", data.shape)
print("First 5 rows of data:")
print(data.head())

# Step 3: Data Preparation


# Selecting relevant columns for classification
selected_columns = ['name', 'mpg', 'cylinders', 'displacement','horsepower', 'weight', 'accelaration', 'model_year', 'origin']
data = data[selected_columns]

# Dropping rows with missing values
data = data.dropna()

# Convert target labels to numerical values if necessary

# Step 4: Split Data into Training and Test Sets
X = data['text']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
# Load pre-trained Transformer-XL model and tokenizer
model_name = 'transfo-xl-wt103'
tokenizer = TransfoXLTokenizer.from_pretrained(model_name)
model = TransfoXLForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Assuming binary classification

# Tokenize input text
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

# Convert labels to tensors
train_labels = torch.tensor(list(y_train))
test_labels = torch.tensor(list(y_test))

# Create data loaders
train_dataset = TensorDataset(train_encodings.input_ids, train_encodings.attention_mask, train_labels)
test_dataset = TensorDataset(test_encodings.input_ids, test_encodings.attention_mask, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Set device to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
model.train()
for epoch in range(5):  # Change the number of epochs as needed
    total_loss = 0
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{5} - Average Loss: {avg_loss:.4f}")


# Step 6: Model Evaluation
model.eval()
total_correct = 0
total_samples = 0
for batch in test_loader:
    batch = tuple(t.to(device) for t in batch)
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    predicted_labels = torch.argmax(logits, dim=1)
    total_correct += (predicted_labels == inputs['labels']).sum().item()
    total_samples += len(inputs['labels'])

accuracy = total_correct / total_samples
print("Model Accuracy:", accuracy)