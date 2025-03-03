class Config:
    # Model configuration
    EMBEDDING_DIM = 64
    NUM_GNN_LAYERS = 2
    
    # Matching configuration
    MIN_SIMILARITY_SCORE = 0.50
    MAX_RECOMMENDATIONS = 5
    
    # Real-world dynamic parameters
    LOCATION_WEIGHT = 0.3        # Higher for location-based matching
    EXPERIENCE_MATCH_WEIGHT = 0.2  # Weight for riding experience matching
    INTERESTS_WEIGHT = 0.3       # Weight for shared interests
    RIDING_STYLE_WEIGHT = 0.2    # Weight for riding style matching

    # Feature flags
    USE_LOCATION_MATCHING = True  # Set to False if you don't have location data
    USE_EXPERIENCE_MATCHING = True
    
    # System settings
    RANDOM_SEED = 42
