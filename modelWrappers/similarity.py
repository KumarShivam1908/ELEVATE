import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class SimilarityEngine:
    """Engine for calculating user similarities and making recommendations"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_data(self, G, user_to_idx):
        """Convert NetworkX graph to PyTorch tensors"""
        # Extract node features and create node feature matrix
        num_nodes = G.number_of_nodes()
        node_features = np.zeros((num_nodes, len(next(iter(G.nodes(data=True)))[1]['features'])))
        
        for i in range(num_nodes):
            node_features[i] = G.nodes[i]['features']
        
        # Extract edges and weights
        edges = []
        edge_weights = []
        
        for u, v, data in G.edges(data=True):
            edges.append([u, v])
            edge_weights.append(data.get('weight', 1.0))
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        
        # Handle empty edge case
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
            edge_weight = torch.tensor(edge_weights, dtype=torch.float32).to(self.device)
        else:
            # Create a self-loop for each node if no edges exist
            self_loops = [[i, i] for i in range(num_nodes)]
            edge_index = torch.tensor(self_loops, dtype=torch.long).t().contiguous().to(self.device)
            edge_weight = torch.ones(len(self_loops), dtype=torch.float32).to(self.device)
        
        return x, edge_index, edge_weight
    
    def get_user_embeddings(self, G, user_to_idx):
        """Get embeddings for all users using the GNN model"""
        x, edge_index, edge_weight = self.prepare_data(G, user_to_idx)
        embeddings = self.model.get_embeddings(x, edge_index, edge_weight)
        
        # Create a mapping from user_id to embedding
        user_embeddings = {}
        for user_id, idx in user_to_idx.items():
            user_embeddings[user_id] = embeddings[idx]
        
        return user_embeddings
    
    def find_matches(self, user_id, users, user_embeddings, top_k=5):
        """Find top-k matches for a given user"""
        target_embedding = user_embeddings[user_id]
        
        similarities = []
        for other_id, other_embedding in user_embeddings.items():
            if other_id == user_id:  # Skip self
                continue
                
            # Calculate cosine similarity
            similarity = F.cosine_similarity(target_embedding, other_embedding, dim=0).item()
            
            # Apply dynamic adjustments based on real-world factors
            adjusted_similarity = self._apply_adjustments(
                similarity, 
                next(u for u in users if u["user_id"] == user_id),  # Get target user
                next(u for u in users if u["user_id"] == other_id)  # Get other user
            )
            
            # Filter out low similarity matches
            if adjusted_similarity >= self.config.MIN_SIMILARITY_SCORE:
                similarities.append({
                    "user_id": other_id,
                    "similarity": adjusted_similarity,
                    "user_data": next(u for u in users if u["user_id"] == other_id)
                })
        
        # Sort by similarity (descending) and take top-k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def _apply_adjustments(self, base_similarity, user1, user2):
        """Apply real-world dynamic adjustments to similarity scores"""
        similarity = base_similarity
        
        # Location adjustment - INCREASED WEIGHT
        if self.config.USE_LOCATION_MATCHING:
            if user1.get("location") == user2.get("location"):
                similarity += self.config.LOCATION_WEIGHT * 0.15  # Increased from 0.1
            # Added: Cities in same region get partial boost
            elif self._are_cities_nearby(user1.get("location", ""), user2.get("location", "")):
                similarity += self.config.LOCATION_WEIGHT * 0.08
        
        # Experience matching adjustment
        if self.config.USE_EXPERIENCE_MATCHING:
            exp1 = user1.get("riding_experience_years", 0)
            exp2 = user2.get("riding_experience_years", 0)
            
            # Similar experience (within 1 year)
            if abs(exp1 - exp2) <= 1:
                similarity += self.config.EXPERIENCE_MATCH_WEIGHT * 0.1
            
            # Potential mentor relationship (experienced rider + newer rider)
            elif exp1 > 3 and exp2 < 2:
                similarity += self.config.EXPERIENCE_MATCH_WEIGHT * 0.05
            elif exp2 > 3 and exp1 < 2:
                similarity += self.config.EXPERIENCE_MATCH_WEIGHT * 0.05
        
        # Shared interests adjustment
        interests1 = set(user1.get("interests", []))
        interests2 = set(user2.get("interests", []))
        shared_interests = interests1.intersection(interests2)
        
        if shared_interests:
            # Bonus based on number of shared interests (with diminishing returns)
            similarity += min(self.config.INTERESTS_WEIGHT * 0.15, 
                             self.config.INTERESTS_WEIGHT * 0.05 * len(shared_interests))
        
        # Riding style compatibility
        styles1 = set(user1.get("riding_style", []))
        styles2 = set(user2.get("riding_style", []))
        shared_styles = styles1.intersection(styles2)
        
        if shared_styles:
            similarity += min(self.config.RIDING_STYLE_WEIGHT * 0.1, 
                             self.config.RIDING_STYLE_WEIGHT * 0.05 * len(shared_styles))
        
        # NEW: Performance mode preference compatibility
        if user1.get("performance_mode_preference") == user2.get("performance_mode_preference"):
            similarity += 0.05
        
        # NEW: Group memberships overlap
        groups1 = set(user1.get("group_memberships", []))
        groups2 = set(user2.get("group_memberships", []))
        shared_groups = groups1.intersection(groups2)
        
        if shared_groups:
            similarity += min(0.08, 0.04 * len(shared_groups))
        
        # NEW: Similar total distance ridden (within 2000km)
        km1 = user1.get("total_km_ridden", 0)
        km2 = user2.get("total_km_ridden", 0)
        if abs(km1 - km2) <= 2000:
            similarity += 0.05
        
        # Cap similarity at 1.0
        return min(1.0, similarity)
    
    def _are_cities_nearby(self, location1, location2):
        """Check if two locations are in the same general region"""
        # Define regions with nearby cities
        north_india = {"delhi", "chandigarh", "dehradun"}
        south_india = {"bangalore", "chennai", "hyderabad", "ooty"}
        west_india = {"mumbai", "pune"}
        east_india = {"shillong", "kolkata"}
        
        # Check if both locations are in the same region
        if location1 in north_india and location2 in north_india:
            return True
        if location1 in south_india and location2 in south_india:
            return True
        if location1 in west_india and location2 in west_india:
            return True
        if location1 in east_india and location2 in east_india:
            return True
        
        return False
    
    def generate_match_explanation(self, user, match):
        """Generate human-readable explanation for why users match"""
        reasons = []
        
        # LOCATION MATCH (prioritized)
        if user.get("location") == match["user_data"].get("location"):
            reasons.append(f"You're both located in {user.get('location').title()}")
        elif self._are_cities_nearby(user.get("location", ""), match["user_data"].get("location", "")):
            reasons.append(f"You're both in the same region ({user.get('location').title()} and {match['user_data'].get('location').title()})")
        
        # Check for same EV model
        if user.get("ev_model") == match["user_data"].get("ev_model"):
            reasons.append(f"You both ride {user.get('ev_model')} motorcycles")
        
        # Check for similar riding experience
        exp1 = user.get("riding_experience_years", 0)
        exp2 = match["user_data"].get("riding_experience_years", 0)
        if abs(exp1 - exp2) <= 1:
            reasons.append("You have similar riding experience")
        elif exp1 > exp2 + 2:
            reasons.append("You could share your extensive riding experience")
        elif exp2 > exp1 + 2:
            reasons.append(f"{match['user_data'].get('name')} has more riding experience and could offer tips")
        
        # Check for shared interests
        interests1 = set(user.get("interests", []))
        interests2 = set(match["user_data"].get("interests", []))
        shared = interests1.intersection(interests2)
        if shared:
            if len(shared) == 1:
                reasons.append(f"You both share an interest in {next(iter(shared)).replace('_', ' ')}")
            elif len(shared) == 2:
                interests_list = [s.replace('_', ' ') for s in shared]
                reasons.append(f"You both share interests in {' and '.join(interests_list)}")
            else:
                sample = [s.replace('_', ' ') for s in list(shared)[:2]]
                reasons.append(f"You share multiple interests including {' and '.join(sample)}")
        
        # Check for shared riding styles
        styles1 = set(user.get("riding_style", []))
        styles2 = set(match["user_data"].get("riding_style", []))
        shared_styles = styles1.intersection(styles2)
        if shared_styles:
            style_list = list(shared_styles)
            if len(style_list) == 1:
                reasons.append(f"You both enjoy {style_list[0].replace('_', ' ')} riding")
            else:
                style_str = ', '.join(s.replace('_', ' ') for s in style_list)
                reasons.append(f"You have compatible riding styles: {style_str}")
        
        # Check for shared routes
        routes1 = set(user.get("favorite_routes", []))
        routes2 = set(match["user_data"].get("favorite_routes", []))
        shared_routes = routes1.intersection(routes2)
        if shared_routes:
            route = next(iter(shared_routes))
            reasons.append(f"You both enjoy riding on {route.replace('_', ' ')}")
        
        # Check for availability match
        avail1 = set(user.get("availability", []))
        avail2 = set(match["user_data"].get("availability", []))
        shared_avail = avail1.intersection(avail2)
        if shared_avail:
            avail = next(iter(shared_avail))
            reasons.append(f"You're both available to ride during {avail.replace('_', ' ')}")
            
        # NEW: Check for performance mode preference match
        if user.get("performance_mode_preference") == match["user_data"].get("performance_mode_preference"):
            mode = user.get("performance_mode_preference", "").replace("_", " ")
            reasons.append(f"You both prefer the {mode} performance mode")
            
        # NEW: Check for shared group memberships
        groups1 = set(user.get("group_memberships", []))
        groups2 = set(match["user_data"].get("group_memberships", []))
        shared_groups = groups1.intersection(groups2)
        if shared_groups:
            group = next(iter(shared_groups))
            reasons.append(f"You're both members of {group.replace('_', ' ')}")
            
        # NEW: Check for similar riding distance
        km1 = user.get("total_km_ridden", 0)
        km2 = match["user_data"].get("total_km_ridden", 0)
        if abs(km1 - km2) <= 2000:
            reasons.append("You have similar riding experience based on kilometers")
        
        # Generate conversation starters
        conversation_starters = []
        
        # EV model conversation starter
        if user.get("ev_model") != match["user_data"].get("ev_model"):
            conversation_starters.append(f"Ask about their experience with the {match['user_data'].get('ev_model')}")
        
        # NEW: Performance mode conversation starter
        if user.get("performance_mode_preference") != match["user_data"].get("performance_mode_preference"):
            other_mode = match["user_data"].get("performance_mode_preference", "").replace("_", " ")
            conversation_starters.append(f"Ask about their experience with the {other_mode} mode")
        
        # Interest conversation starter
        if interests2 - interests1:
            unique_interest = next(iter(interests2 - interests1))
            conversation_starters.append(f"Ask about their interest in {unique_interest.replace('_', ' ')}")
        
        # Route conversation starter
        if routes2 - routes1:
            unique_route = next(iter(routes2 - routes1))
            conversation_starters.append(f"Ask about their experience riding {unique_route.replace('_', ' ')}")
            
        # NEW: Group membership conversation starter
        if groups2 - groups1:
            unique_group = next(iter(groups2 - groups1))
            conversation_starters.append(f"Ask about their experience with {unique_group.replace('_', ' ')}")
            
        # NEW: App features conversation starter
        features1 = set(user.get("app_features_used", []))
        features2 = set(match["user_data"].get("app_features_used", []))
        if features2 - features1:
            unique_feature = next(iter(features2 - features1))
            conversation_starters.append(f"Ask how they use the {unique_feature.replace('_', ' ')} feature")
        
        return {
            "match_reasons": reasons,
            "conversation_starters": conversation_starters,
            "similarity_score": f"{match['similarity'] * 100:.1f}% match"
        }