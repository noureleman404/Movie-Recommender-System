import streamlit as st
import torch
import torch.nn as nn
import pandas as pd

# ----------------------
# Model architecture 
# ----------------------
class CollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(CollaborativeFiltering, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embeddings(user_ids)
        item_embed = self.item_embeddings(item_ids)
        user_bias = self.user_bias(user_ids).squeeze()
        item_bias = self.item_bias(item_ids).squeeze()
        return (user_embed * item_embed).sum(dim=1) + user_bias + item_bias
    
# ----------------------
# Data & Model Loading
# ----------------------
@st.cache_data
def load_data():
    # Load movies
    movies = pd.read_csv("data/u.item", sep="|", encoding="latin-1", 
                         names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL", 
                                "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", 
                                "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", 
                                "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
    
    # Load ratings
    ratings = pd.read_csv("data/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
    
    return movies, ratings

@st.cache_resource
def load_model(num_users, num_items):
    embedding_dim = 50
    model = CollaborativeFiltering(num_users, num_items, embedding_dim)
    # Load the weights saved from your notebook
    model.load_state_dict(torch.load('models/movie_recommender(final).pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# ---------------------
# Recommendation Logic
# ---------------------
def recommend_top_k(model, user_id, all_movie_ids, rated_movie_ids, k=5):
    # Get movies the user HAS NOT rated
    unseen_movie_ids = list(set(all_movie_ids) - set(rated_movie_ids))
    unseen_movie_ids_tensor = torch.tensor(unseen_movie_ids, dtype=torch.long)
    
    # Predict ratings for unseen movies
    user_ids_tensor = torch.full_like(unseen_movie_ids_tensor, user_id)
    with torch.no_grad():
        predictions = model(user_ids_tensor, unseen_movie_ids_tensor)
    
    # Get top k
    _, top_indices = torch.topk(predictions, k=k)
    recommended_ids = unseen_movie_ids_tensor[top_indices].numpy()
    return recommended_ids

# ---------------
# Streamlit UI
# ---------------
st.title("🍿 Movie Recommendation System")
st.write("A Collaborative Filtering system powered by PyTorch.")

# Load data
movies_df, ratings_df = load_data()

# Calculate bounds
num_users = ratings_df['user_id'].max() + 1
num_items = movies_df['movie_id'].max() + 1
all_movie_ids = movies_df['movie_id'].tolist()

# Load model
model = load_model(num_users, num_items)

# UI elements
st.sidebar.header("User Settings")
# Select a user ID from existing users
st.sidebar.write(
    "Each profile represents a unique movie taste learned from historical ratings."
)
selected_user = st.sidebar.selectbox("Choose a Viewer Profile", ratings_df['user_id'].unique())

if st.sidebar.button("Get Recommendations"):
    st.subheader("🎯 Your Personalized Movie Night Picks")

    st.markdown(
        f"""
    Based on this viewer’s past highly rated movies,
    the model predicts these films are most aligned with their taste profile.
    """
    )
    
    # Find movies this user already rated
    rated_movie_ids = ratings_df[ratings_df['user_id'] == selected_user]['movie_id'].tolist()
    
    # Generate predictions
    top_movie_ids = recommend_top_k(model, selected_user, all_movie_ids, rated_movie_ids, k=5)
    
    # Fetch movie titles
    recommended_movies = movies_df[movies_df['movie_id'].isin(top_movie_ids)][['title', 'release_date']]
    
    # Display visually
    for index, row in recommended_movies.iterrows():
        st.success(f"🎬 **{row['title']}**")
        
    st.divider()
    with st.expander("🔎 What shaped this viewer's taste?"):
        st.markdown(
            "These are movies this profile rated 4⭐ or higher. "
            "The model uses this behavioral signal to learn taste patterns."
        )
        user_history = ratings_df[(ratings_df['user_id'] == selected_user) & (ratings_df['rating'] >= 4)]
        history_titles = pd.merge(user_history, movies_df, on='movie_id')[['title', 'rating']]
        st.dataframe(history_titles.sort_values(by='rating', ascending=False), hide_index=True)