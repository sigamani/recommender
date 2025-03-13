import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate mock travel search data
num_users = 1000
num_destinations = 200

# Sample destinations
destinations = [
    "Paris", "New York", "Tokyo", "London", "Bangkok", "Dubai", "Rome", "Istanbul", "Singapore", "Barcelona",
    "Hong Kong", "Los Angeles", "Sydney", "Las Vegas", "Berlin", "Chicago", "San Francisco", "Madrid", "Moscow", "Toronto"
]

# Generate user search history
user_ids = np.random.randint(10000, 99999, size=num_users)
searched_destinations = np.random.choice(destinations, size=num_users)
search_dates = pd.date_range(start="2024-01-01", periods=num_users, freq="D")
search_preferences = np.random.choice(["beach", "city", "mountains", "cultural", "adventure"], size=num_users)
search_frequencies = np.random.randint(1, 10, size=num_users)  # How often they search for travel

# Generate embeddings (random for now, but structured)
embedding_dim = 128  # Size of the vector embeddings
destination_embeddings = {dest: np.random.rand(embedding_dim) for dest in destinations}

# Create DataFrame
df = pd.DataFrame({
    "user_id": user_ids,
    "searched_destination": searched_destinations,
    "search_date": search_dates,
    "search_preference": search_preferences,
    "search_frequency": search_frequencies,
})

# Save dataset
csv_path = "/mnt/data/travel_search_data.csv"
df.to_csv(csv_path, index=False)

# Convert embeddings to a DataFrame for vector search
embeddings_df = pd.DataFrame.from_dict(destination_embeddings, orient="index")
embeddings_df.insert(0, "destination", embeddings_df.index)
embeddings_csv_path = "/mnt/data/destination_embeddings.csv"
embeddings_df.to_csv(embeddings_csv_path, index=False)

# Display DataFrame for review
import ace_tools as tools
tools.display_dataframe_to_user(name="Mock Travel Search Data", dataframe=df)

csv_path, embeddings_csv_path
