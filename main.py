import torch
import numpy as np
import random
from data.users import get_sample_users  # Changed import to correct file
from data.data_processor import DataProcessor
from modelWrappers.gnn import SimpleGNN
from modelWrappers.similarity import SimilarityEngine
from utils.visualization import visualize_user_graph, visualize_embeddings, display_match_results
from config import Config
from utils.json_response import get_matches_json

def set_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    # Set up configuration and seed
    config = Config()
    set_seeds(config.RANDOM_SEED)
    
    # Load user data
    users = get_sample_users()
    print(f"Loaded {len(users)} users")
    
    # Process data into graph
    processor = DataProcessor(config)
    G, user_to_idx = processor.create_user_graph(users)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Visualize the initial graph (optional)
    visualize_user_graph(G, user_to_idx, "EV Motorcycle Riders Network")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the GNN model
    sample_features = processor.extract_node_features(users[0])
    input_dim = len(sample_features)
    model = SimpleGNN(
        input_dim=input_dim,
        hidden_dim=32,
        embedding_dim=config.EMBEDDING_DIM,
        num_layers=config.NUM_GNN_LAYERS
    ).to(device)
    print(f"Initialized GNN model with input dimension: {input_dim}")
    
    # Create similarity engine
    similarity_engine = SimilarityEngine(model, config)
    
    # Get user embeddings
    user_embeddings = similarity_engine.get_user_embeddings(G, user_to_idx)
    print("Generated user embeddings")
    
    # Visualize embeddings (optional)
    visualize_embeddings(user_embeddings, users, "EV Motorcycle Riders Embedding Space")
    
    # Find matches for each user
    for user in users:
        matches = similarity_engine.find_matches(
            user["user_id"], users, user_embeddings, top_k=config.MAX_RECOMMENDATIONS
        )
        
        # Generate match explanations
        match_explanations = [
            similarity_engine.generate_match_explanation(user, match)
            for match in matches
        ]
        
        # Display results
        if matches:
            display_match_results(user, matches, match_explanations)
        else:
            print(f"\nNo suitable matches found for {user['name']} ({user['user_id']})")
        
        print("JSON test:")
        # Example usage in main.py

        # Find matches and get JSON for a specific user
        user_id = "user5"  # Replace with actual user ID
        user_name = "Vikram Singh"  # Replace with actual user name
        print(f"{user_id}:{user_name}")
        json_result = get_matches_json(
            user_name, 
            user_id, 
            users, 
            user_embeddings, 
            similarity_engine, 
            max_recommendations=config.MAX_RECOMMENDATIONS
        )

        # Write to file or use the JSON string as needed
        with open(f"matches_{user_id}.json", "w") as f:
            f.write(json_result)

if __name__ == "__main__":
    main()