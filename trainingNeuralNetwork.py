# 4. NN Training
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np
import random
from UserProfileRecommender import UserProfileRecommender  # Import your model

# ------------------------- #
# 1. Ensure Reproducibility #
# ------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# If using MPS (Apple GPU), set MPS seed
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

# ------------------------ #
# 2. Detect Device (MPS)   #
# ------------------------ #
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------ #
# 3. Load Training Data    #
# ------------------------ #
df = pd.read_csv("user_post_interactions.csv")

# Determine number of users and posts
num_users = df["user_id"].max() + 1
num_posts = df["post_id"].max() + 1

# Convert data to PyTorch tensors and move them to the selected device
user_tensor = torch.tensor(df["user_id"].values, dtype=torch.long).to(device)
post_tensor = torch.tensor(df["post_id"].values, dtype=torch.long).to(device)
interaction_tensor = torch.tensor(df["interaction"].values, dtype=torch.float32).to(device)

# Create DataLoader for batching and shuffling
dataset = data.TensorDataset(user_tensor, post_tensor, interaction_tensor)
dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True)  # Increased batch size for efficiency

# --------------------------- #
# 4. Initialize Model & Weights #
# --------------------------- #
model = UserProfileRecommender(num_users, num_posts).to(device)

# Custom weight initialization (Xavier Initialization)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)  # Apply weight initialization

# ---------------------------- #
# 5. Define Loss & Optimizer   #
# ---------------------------- #
criterion = nn.BCELoss().to(device)  # Binary cross-entropy loss
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)  # AdamW with weight decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)  # Reduce LR every 2 epochs

# ------------------------ #
# 6. Training Loop         #
# ------------------------ #
num_epochs = 50  # Increased number of epochs
best_loss = float("inf")  # Track best loss for model saving

for epoch in range(num_epochs):
    model.train()  # Set to training mode

    epoch_loss = 0
    for user, post, interaction in dataloader:
        # Move batch data to the correct device
        user, post, interaction = user.to(device), post.to(device), interaction.to(device)

       # Lower Learning Rate for More Precise Learning
       # optimizer.zero_grad()  # Reset gradients
        predictions = model(user, post).squeeze()  # Forward pass
        loss = criterion(predictions, interaction)  # Compute loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        epoch_loss += loss.item()  # Accumulate loss

    scheduler.step()  # Adjust learning rate

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # ---------------------- #
    # 7. Save Best Model    #
    # ---------------------- #
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "user_profile_recommender_best.pth")
        #print("✅ Model saved (Best so far)")

# ------------------------ #
# 8. Load & Evaluate Model #
# ------------------------ #
model.load_state_dict(torch.load("user_profile_recommender_best.pth"))
model.to(device)
model.eval()  # Set to evaluation mode
#print("🔄 Best trained model loaded for evaluation!")

# Confirm whether MPS was used
if torch.backends.mps.is_available():
    print("🚀 Training completed using MPS (Metal Performance Shaders) on Apple Silicon.")
else:
    print("⚠️ Training completed using CPU.")

