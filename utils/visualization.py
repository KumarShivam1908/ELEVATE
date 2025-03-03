import matplotlib.pyplot as plt
import networkx as nx
import torch
from sklearn.decomposition import PCA
import numpy as np

def visualize_user_graph(G, user_to_idx, title="User Similarity Graph"):
    """Visualize the user graph"""
    plt.figure(figsize=(10, 8))
    
    # Create a mapping from index to user_id for labeling
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    
    # Get node positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7)
    
    # Add labels
    labels = {idx: idx_to_user[idx] for idx in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_embeddings(user_embeddings, users, title="User Embedding Visualization"):
    """Visualize user embeddings in 2D using PCA"""
    # Extract embeddings and users
    user_ids = list(user_embeddings.keys())
    embeddings = torch.stack([user_embeddings[uid] for uid in user_ids]).cpu().numpy()
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create a mapping for coloring by EV model
    ev_models = {}
    for user in users:
        ev_models[user["user_id"]] = user.get("ev_model", "Unknown")
    
    unique_models = list(set(ev_models.values()))
    model_to_color = {model: plt.cm.tab10(i) for i, model in enumerate(unique_models)}
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    for i, user_id in enumerate(user_ids):
        model = ev_models[user_id]
        plt.scatter(
            embeddings_2d[i, 0], 
            embeddings_2d[i, 1], 
            color=model_to_color[model], 
            s=100, 
            alpha=0.7
        )
        plt.text(
            embeddings_2d[i, 0] + 0.01, 
            embeddings_2d[i, 1] + 0.01, 
            user_id, 
            fontsize=9
        )
    
    # Add legend
    for model, color in model_to_color.items():
        plt.scatter([], [], color=color, label=model)
    plt.legend(title="EV Model", loc="best")
    
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def display_match_results(user, matches, match_explanations):
    """Display match results in a readable format"""
    print(f"\n===== MATCHES FOR {user['name']} ({user['user_id']}) =====")
    
    for i, match in enumerate(matches):
        explanation = match_explanations[i]
        print(f"\n{i+1}. {match['user_data']['name']} - {explanation['similarity_score']}")
        print("   EV Model:", match['user_data']['ev_model'])
        print("   Experience:", f"{match['user_data']['riding_experience_years']} years")
        
        print("\n   Why you match:")
        for reason in explanation['match_reasons']:
            print(f"   • {reason}")
        
        print("\n   Conversation starters:")
        for starter in explanation['conversation_starters']:
            print(f"   • {starter}")
        
        print("\n   -------------------------------------------")