#2. Model Architecture

import torch
import torch.nn as nn
import torch.optim as optim

class UserProfileRecommender(nn.Module):
    def __init__(self, num_users, num_posts, embedding_dim=64):
        super(UserProfileRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.post_embedding = nn.Embedding(num_posts, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, post_ids):
        user_emb = self.user_embedding(user_ids)
        post_emb = self.post_embedding(post_ids)
        x = torch.cat([user_emb, post_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x