#3. Simulate user Interaction data

import pandas as pd
import numpy as np

# Simulate user interactions with posts
users = np.arange(1, 101)  # 100 users
posts = np.arange(1, 501)  # 500 posts
data = []

for user in users:
    for post in np.random.choice(posts, size=np.random.randint(5, 20), replace=False):
        interaction = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% chance of liking a post
        data.append([user, post, interaction])

df = pd.DataFrame(data, columns=["user_id", "post_id", "interaction"])
df.to_csv("user_post_interactions.csv", index=False)
