import networkx as nx
import torch
from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

def visualize_user_graph(G, user_to_idx, title="User Similarity Graph"):
    """Visualize the user graph using Plotly"""
    # Create a mapping from index to user_id for labeling
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    
    # Get node positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Extract node positions
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"ID: {idx_to_user[node]}")
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            color='lightblue',
            size=15,
            line=dict(width=2, color='DarkSlateGrey')))
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=title,
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    fig.show()

def visualize_embeddings(user_embeddings, users, title="User Embedding Visualization"):
    """Visualize user embeddings in 2D using PCA and Plotly"""
    # Extract embeddings and users
    user_ids = list(user_embeddings.keys())
    embeddings = torch.stack([user_embeddings[uid] for uid in user_ids]).cpu().numpy()
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create dataframe for plotting
    plot_data = []
    for i, user_id in enumerate(user_ids):
        # Find the user data
        user_data = next((u for u in users if u["user_id"] == user_id), None)
        if user_data:
            plot_data.append({
                'x': embeddings_2d[i, 0],
                'y': embeddings_2d[i, 1],
                'user_id': user_id,
                'name': user_data.get('name', ''),
                'ev_model': user_data.get('ev_model', 'Unknown'),
                'experience': user_data.get('riding_experience_years', 0),
                'location': user_data.get('location', 'Unknown'),
                'riding_style': ', '.join(user_data.get('riding_style', [])),
                'interests': ', '.join(user_data.get('interests', []))
            })
    
    # Create figure using plotly express
    fig = px.scatter(
        plot_data, 
        x='x', 
        y='y', 
        color='ev_model',
        hover_data=['name', 'experience', 'location', 'riding_style', 'interests'],
        title=title,
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    # Add annotations for each point
    annotations = []
    for point in plot_data:
        annotations.append(dict(
            x=point['x'],
            y=point['y'],
            text=point['user_id'],
            showarrow=False,
            font=dict(size=10),
            xshift=10,
            yshift=10
        ))
    
    fig.update_layout(
        annotations=annotations,
        legend_title_text='EV Model',
        height=700,
        width=900,
        hovermode='closest'
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
    
    fig.show()
def visualize_user_graph(G, user_to_idx, title="User Similarity Graph"):
    """Visualize the user graph using Plotly"""
    # Create a mapping from index to user_id for labeling
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    
    # Get node positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Extract node positions
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"ID: {idx_to_user[node]}")
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            color='lightblue',
            size=15,
            line=dict(width=2, color='DarkSlateGrey')))
    
    # Create the figure with updated title formatting
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict(
                        text=title,
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    fig.show()

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

def visualize_match_results(user, matches, match_explanations):
    """Visualize match results using an interactive Plotly chart"""
    # Prepare data for visualization
    match_data = []
    for i, match in enumerate(matches):
        explanation = match_explanations[i]
        similarity = float(explanation['similarity_score'].strip('% match'))
        
        match_data.append({
            'rank': i + 1,
            'name': match['user_data']['name'],
            'user_id': match['user_data']['user_id'],
            'ev_model': match['user_data']['ev_model'],
            'similarity': similarity,
            'experience': match['user_data']['riding_experience_years'],
            'location': match['user_data'].get('location', '').replace('_', ' ').title(),
            'match_reasons': '<br>• ' + '<br>• '.join(explanation['match_reasons']),
            'conversation_starters': '<br>• ' + '<br>• '.join(explanation['conversation_starters']),
        })
    
    # Create the figure
    fig = go.Figure()
    
    # Add bar for each match
    for item in match_data:
        fig.add_trace(
            go.Bar(
                x=[item['similarity']],
                y=[item['name']],
                orientation='h',
                name=item['name'],
                text=f"{item['similarity']:.1f}%",
                textposition='auto',
                hovertext=f"<b>{item['name']}</b> ({item['user_id']})<br><br>" +
                          f"<b>EV Model:</b> {item['ev_model']}<br>" +
                          f"<b>Experience:</b> {item['experience']} years<br>" +
                          f"<b>Location:</b> {item['location']}<br><br>" +
                          f"<b>Match Reasons:</b>{item['match_reasons']}<br><br>" +
                          f"<b>Conversation Starters:</b>{item['conversation_starters']}",
                hoverinfo='text',
                marker=dict(
                    color=item['similarity'],
                    colorscale='Viridis',
                    cmin=60,
                    cmax=100,
                    line=dict(width=1)
                )
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"Match Results for {user['name']} ({user['user_id']})",
        xaxis=dict(
            title='Match Percentage',
            range=[0, 100]
        ),
        yaxis=dict(
            title='Recommended Riders',
            categoryorder='total ascending'
        ),
        height=400 + (len(match_data) * 40),
        width=900,
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    fig.show()

def get_matches_json(user_name, user_id, users, user_embeddings, similarity_engine, max_recommendations=5):
    """
    Get matches for a specific user and return them in JSON format
    
    Args:
        user_name (str): Name of the user to find matches for
        user_id (str): ID of the user to find matches for
        users (list): List of all user dictionaries
        user_embeddings (dict): Dictionary mapping user IDs to embeddings
        similarity_engine (SimilarityEngine): Instance of the similarity engine
        max_recommendations (int, optional): Maximum number of recommendations to return
    
    Returns:
        str: JSON string containing the match data
    """
    # Find the user in the users list
    user = next((u for u in users if u['user_id'] == user_id), None)
    if not user:
        return json.dumps({
            "user_id": user_id,
            "user_name": user_name,
            "error": "User not found",
            "matches": []
        })
    
    # Find matches for the user
    matches = similarity_engine.find_matches(
        user_id, users, user_embeddings, top_k=max_recommendations
    )
    
    # Generate match explanations
    match_explanations = [
        similarity_engine.generate_match_explanation(user, match)
        for match in matches
    ]
    
    # Build the result dictionary
    result = {
        "user_id": user_id,
        "user_name": user_name,
        "total_matches": len(matches),
        "matches": []
    }
    
    # Add each match with its details
    for i, match in enumerate(matches):
        explanation = match_explanations[i]
        match_data = {
            "rank": i + 1,
            "rider_name": match['user_data']['name'],
            "rider_id": match['user_data']['user_id'],
            "ev_model": match['user_data']['ev_model'],
            "experience_years": match['user_data']['riding_experience_years'],
            "similarity_score": explanation['similarity_score'],
            "match_reasons": explanation['match_reasons'],
            "conversation_starters": explanation['conversation_starters']
        }
        result["matches"].append(match_data)
    
    return json.dumps(result, indent=2)