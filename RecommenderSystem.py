#5 run the system loading trained model
import torch
import argparse
from UserProfileRecommender import UserProfileRecommender  # Import trained model

class RecommenderSystem:
    def __init__(self, model_path, num_users, num_posts):
        """
        Initialize the recommender system by loading the pre-trained model.

        Args:
            model_path (str): Path to the saved model file (.pth)
            num_users (int): Number of users in the dataset
            num_posts (int): Number of posts in the dataset
        """
        # Detect device (MPS for Mac, CPU fallback)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize the model
        self.model = UserProfileRecommender(num_users, num_posts)
        
        # Load the trained model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        print("✅ Model loaded successfully!")

    def predict_interaction(self, user_id, post_id):
        """
        Predict the interaction likelihood for a given user and post.

        Args:
            user_id (int): The ID of the user
            post_id (int): The ID of the post

        Returns:
            float: The predicted interaction score (0 to 1)
        """
        user_tensor = torch.tensor([user_id], dtype=torch.long).to(self.device)
        post_tensor = torch.tensor([post_id], dtype=torch.long).to(self.device)

        with torch.no_grad():  # Disable gradient calculation for inference
            prediction = self.model(user_tensor, post_tensor)

        return prediction.item()

# ---------------------- #
# ✅ Command Line Interface #
# ---------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict user-post interaction score.")
    parser.add_argument("user_id", type=int, help="User ID")
    parser.add_argument("post_id", type=int, help="Post ID")

    args = parser.parse_args()

    # Define dataset size (replace with actual numbers used during training)
    NUM_USERS = 101  # Update based on your dataset
    NUM_POSTS = 501  # Update based on your dataset

    MODEL_PATH = "/Users/hamidporasl/python/AI/user_profile_recommender_best.pth"  # Path to trained model

    # If you want to save the model, do it when you have a trained model ready
    # You might do something like this within your training code:
    # torch.save(model.state_dict(), MODEL_PATH)
    # print(f"✅ Model saved successfully at {MODEL_PATH}!")

    # Initialize recommender system
    recommender = RecommenderSystem(MODEL_PATH, NUM_USERS, NUM_POSTS)

    # Predict interaction score
    predicted_score = recommender.predict_interaction(args.user_id, args.post_id)
    print(f"Predicted interaction score for User {args.user_id} and Post {args.post_id}: {predicted_score:.4f}")
