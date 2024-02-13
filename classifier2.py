import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

# Define your neural network architecture (net) and dataset (train_set) here
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100, 50)  # Assuming input dimension is 100
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Convert DataLoader to DataLoader to be used with Hugging Face Trainer
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)

# Define training arguments for the Trainer
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=32,
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_total_limit=1,           # Save only the best checkpoint
    save_steps=0,                 # Save checkpoints only at the end of each epoch
    no_cuda=True if device == "cpu" else False,  # Set True if using CPU
    logging_dir="./logs",
    logging_steps=2000,           # Print loss every 2000 steps
    evaluation_steps=2000,        # Evaluate every 2000 steps
    save_on_each_node=False,      # Set True if using distributed training
    load_best_model_at_end=True   # Load the best checkpoint at the end of training
)

# Define a custom training loop function
def custom_training_loop(model, train_dataloader, optimizer, criterion, device):
    model.train()
    losses = []
    
    for epoch in range(training_args.num_train_epochs):
        running_loss = 0.0
        
        for batch in tqdm(train_dataloader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        losses.append(running_loss)
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader)}")

# Instantiate your model here
# Instantiate early stopping callback
early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)

# Start the training loop using the Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None,   # Pass your training dataset if needed
    data_collator=None,   # Pass your data collator if needed
    callbacks=[early_stopping]
)

# Train the model using the Trainer
trainer.train()

print("Training complete!")
