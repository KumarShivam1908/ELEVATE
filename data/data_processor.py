import numpy as np
import networkx as nx

class DataProcessor:
    """Processes user data into a format suitable for GNN"""
    
    def __init__(self, config):
        self.config = config
        
        # Define feature categories
        self.ev_models = ["Zero SR/F", "Zero SR/S", "Zero FX", "LiveWire One", "Energica Eva", "Other"]
        self.riding_styles = ["commuting", "touring", "sport_riding", "off_road", "city_rides", "weekend_trips", "learning"]
        self.locations = ["san_francisco", "oakland", "san_jose", "berkeley", "other"]
        self.interests = ["sustainable tech", "photography", "camping", "mechanics", "coffee", 
                        "new_tech", "learning", "gaming", "racing", "travel", "adventure"]
        self.availabilities = ["weekday_morning", "weekday_evening", "weekend_morning", 
                             "weekend_afternoon", "weekend_all_day"]
    
    def extract_node_features(self, user):
        """Extract features from a user into a numerical vector"""
        features = []
        
        # EV Model (one-hot encoding)
        model_features = [1.0 if user.get("ev_model") == model else 0.0 for model in self.ev_models]
        features.extend(model_features)
        
        # Riding Experience (normalized)
        exp_years = user.get("riding_experience_years", 0)
        # Normalize to 0-1 range (assuming max experience of 10 years)
        features.append(min(1.0, exp_years / 10.0))
        
        # Riding Style (multi-hot encoding)
        user_styles = user.get("riding_style", [])
        style_features = [1.0 if style in user_styles else 0.0 for style in self.riding_styles]
        features.extend(style_features)
        
        # Location (one-hot encoding) - only if location matching is enabled
        if self.config.USE_LOCATION_MATCHING:
            user_location = user.get("location", "other")
            if user_location not in self.locations:
                user_location = "other"
            location_features = [1.0 if user_location == loc else 0.0 for loc in self.locations]
            features.extend(location_features)
        
        # Interests (multi-hot encoding)
        user_interests = user.get("interests", [])
        interest_features = [1.0 if interest in user_interests else 0.0 for interest in self.interests]
        features.extend(interest_features)
        
        # Availability (multi-hot encoding)
        user_availability = user.get("availability", [])
        availability_features = [1.0 if time in user_availability else 0.0 for time in self.availabilities]
        features.extend(availability_features)
        
        return np.array(features, dtype=np.float32)
    
    def create_user_graph(self, users):
        """Create a graph representation of users and their connections"""
        G = nx.Graph()
        
        # Map user IDs to node indices
        user_to_idx = {user["user_id"]: i for i, user in enumerate(users)}
        
        # Add nodes with features
        for user in users:
            node_idx = user_to_idx[user["user_id"]]
            node_features = self.extract_node_features(user)
            G.add_node(node_idx, features=node_features, user_id=user["user_id"])
        
        # Add edges based on existing connections (group memberships)
        groups_to_users = {}
        for user in users:
            user_idx = user_to_idx[user["user_id"]]
            user_groups = user.get("group_memberships", [])
            
            for group in user_groups:
                if group not in groups_to_users:
                    groups_to_users[group] = []
                groups_to_users[group].append(user_idx)
        
        # Create edges between users in the same groups
        for group, members in groups_to_users.items():
            for i, user1_idx in enumerate(members):
                for user2_idx in members[i+1:]:
                    # Add edge with weight 0.5 (this can be adjusted)
                    if G.has_edge(user1_idx, user2_idx):
                        # Increase weight if already connected
                        G[user1_idx][user2_idx]['weight'] += 0.2
                    else:
                        G.add_edge(user1_idx, user2_idx, weight=0.5)
        
        # Add edges based on shared routes
        routes_to_users = {}
        for user in users:
            user_idx = user_to_idx[user["user_id"]]
            user_routes = user.get("favorite_routes", [])
            
            for route in user_routes:
                if route not in routes_to_users:
                    routes_to_users[route] = []
                routes_to_users[route].append(user_idx)
        
        # Create edges between users with shared routes
        for route, riders in routes_to_users.items():
            for i, user1_idx in enumerate(riders):
                for user2_idx in riders[i+1:]:
                    # Add edge with weight 0.3 (route sharing less strong than group membership)
                    if G.has_edge(user1_idx, user2_idx):
                        G[user1_idx][user2_idx]['weight'] += 0.3
                    else:
                        G.add_edge(user1_idx, user2_idx, weight=0.3)
        
        # Add edges based on shared charging stations
        stations_to_users = {}
        for user in users:
            user_idx = user_to_idx[user["user_id"]]
            user_stations = user.get("charging_stations", [])
            
            for station in user_stations:
                if station not in stations_to_users:
                    stations_to_users[station] = []
                stations_to_users[station].append(user_idx)
        
        # Create edges between users with shared charging stations
        for station, users_idx in stations_to_users.items():
            for i, user1_idx in enumerate(users_idx):
                for user2_idx in users_idx[i+1:]:
                    # Add edge with weight 0.2 (charging station less strong than routes)
                    if G.has_edge(user1_idx, user2_idx):
                        G[user1_idx][user2_idx]['weight'] += 0.2
                    else:
                        G.add_edge(user1_idx, user2_idx, weight=0.2)
        
        return G, user_to_idx
