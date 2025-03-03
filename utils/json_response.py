import matplotlib.pyplot as plt
import networkx as nx
import torch
from sklearn.decomposition import PCA
import numpy as np
import json
from datetime import datetime


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
        user_id, users, user_embeddings, top_k=max_recommendations * 2  # Get more matches initially to filter
    )
    
    # Filter matches based on location if enabled
    filtered_matches = []
    for match in matches:
        # Skip riders from very different locations if they're far apart
        # This is a simplistic check - you could implement a more sophisticated distance calculation
        if user.get('location') != match['user_data'].get('location'):
            # Lower the similarity score for distant matches
            match['similarity'] *= 0.8  # Reduce score by 20%
        
        # Keep the match in the filtered list regardless (just with adjusted score)
        filtered_matches.append(match)
    
    # Re-sort after adjustment and take top recommendations
    filtered_matches.sort(key=lambda x: x["similarity"], reverse=True)
    final_matches = filtered_matches[:max_recommendations]
    
    # Generate match explanations
    match_explanations = [
        similarity_engine.generate_match_explanation(user, match)
        for match in final_matches
    ]
    
    # Build the result dictionary
    result = {
        "user_id": user_id,
        "user_name": user_name,
        "user_location": user.get("location", "Unknown"),
        "user_ev_model": user.get("ev_model", "Unknown"),
        "user_riding_experience": user.get("riding_experience_years", 0),
        "user_riding_style": user.get("riding_style", []),
        "total_matches": len(final_matches),
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
        "matches": []
    }
    
    # Add each match with its details
    for i, match in enumerate(final_matches):
        explanation = match_explanations[i]
        match_user = match['user_data']
        
        # Calculate compatibility factors
        same_location = user.get('location') == match_user.get('location')
        same_model_family = user.get('ev_model', '').split()[0] == match_user.get('ev_model', '').split()[0]
        shared_routes = set(user.get('favorite_routes', [])).intersection(set(match_user.get('favorite_routes', [])))
        shared_availability = set(user.get('availability', [])).intersection(set(match_user.get('availability', [])))
        
        match_data = {
            "rank": i + 1,
            "rider_name": match_user['name'],
            "rider_id": match_user['user_id'],
            "ev_model": match_user['ev_model'],
            "experience_years": match_user['riding_experience_years'],
            "location": match_user.get('location', 'Unknown'),
            "performance_mode_preference": match_user.get('performance_mode_preference', 'Unknown'),
            "total_km_ridden": match_user.get('total_km_ridden', 0),
            "riding_style": match_user.get('riding_style', []),
            "favorite_routes": match_user.get('favorite_routes', []),
            "availability": match_user.get('availability', []),
            "app_features_used": match_user.get('app_features_used', []),
            "interests": match_user.get('interests', []),
            "similarity_score": explanation['similarity_score'],
            "match_reasons": explanation['match_reasons'],
            "conversation_starters": explanation['conversation_starters'],
            "compatibility_factors": {
                "same_location": same_location,
                "same_model_family": same_model_family,
                "shared_routes": list(shared_routes),
                "shared_availability": list(shared_availability)
            }
        }
        result["matches"].append(match_data)
    
    return json.dumps(result, indent=2)

def display_match_results(user, matches, match_explanations):
    """Display match results in a readable format"""
    print(f"\n===== MATCHES FOR {user['name']} ({user['user_id']}) =====")
    print(f"Location: {user.get('location', 'Unknown')}")
    print(f"EV Model: {user.get('ev_model', 'Unknown')}")
    print(f"Experience: {user.get('riding_experience_years', 0)} years")
    
    for i, match in enumerate(matches):
        explanation = match_explanations[i]
        match_data = match['user_data']
        
        print(f"\n{i+1}. {match_data['name']} - {explanation['similarity_score']}")
        print(f"   Location: {match_data.get('location', 'Unknown')}")
        print(f"   EV Model: {match_data['ev_model']}")
        print(f"   Experience: {match_data['riding_experience_years']} years")
        print(f"   Riding Style: {', '.join(match_data.get('riding_style', ['Unknown']))}")
        
        # Show if distance is a factor
        if user.get('location') != match_data.get('location'):
            print(f"   Note: Different city - {user.get('location')} vs {match_data.get('location')}")
        
        # Show shared availability
        shared_avail = set(user.get('availability', [])).intersection(set(match_data.get('availability', [])))
        if shared_avail:
            print(f"   Available Together: {', '.join(shared_avail)}")
        
        print("\n   Why you match:")
        for reason in explanation['match_reasons']:
            print(f"   • {reason}")
        
        print("\n   Conversation starters:")
        for starter in explanation['conversation_starters']:
            print(f"   • {starter}")
        
        print("\n   -------------------------------------------")