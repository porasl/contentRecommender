from fastapi import FastAPI
import torch

app = FastAPI()

@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    user_id_tensor = torch.tensor([user_id], dtype=torch.long)
    post_ids = torch.arange(1, num_posts, dtype=torch.long)
    scores = model(user_id_tensor.expand_as(post_ids), post_ids).detach().numpy()
    
    top_posts = post_ids.numpy()[scores.flatten().argsort()[-5:][::-1]]
    return {"user_id": user_id, "recommended_posts": top_posts.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

