# 🔋 ELEVATE

> **Electric Vehicle Rider Network Platform**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ELEVATE is a sophisticated recommendation system designed to connect electric motorcycle enthusiasts based on their riding preferences, experience levels, and geographic proximity. Focusing on the RapteeHV electric motorcycle community, this platform uses graph neural networks to create meaningful rider connections.

## ✨ Features

- **Smart Rider Matching:** Connects riders based on multiple attributes including location, riding style, experience, and interests
- **Location-Centric Matching:** Prioritizes connections between riders in the same area or region
- **Interactive Visualizations:** Explore rider networks and embedding spaces with intuitive plotly visualizations
- **Conversation Starters:** Generates personalized ice-breakers to help initiate discussions between matched riders
- **JSON API Support:** Flexible match results in JSON format for integration with mobile apps and web services

## 🛠️ Technology Stack

- **Python:** Core language for algorithm implementation
- **PyTorch:** Deep learning framework for neural network implementation
- **PyTorch Geometric:** Extension for graph neural networks

## 📁 Project Structure

```
ELEVATE/
│
├── data/
│   └── users.py           # User data and profile information
│
├── modelWrappers/
│   ├── gnn.py             # Graph Neural Network implementation
│   └── similarity.py      # Similarity calculation and matching engine
│
├── utils/
│   ├── processor.py       # Data processing utilities 
│   └── visualization.py   # Data visualization functions
│
├── config.py              # Configuration settings
└── main.py                # Main application entry point
```

## 🔄 How It Works

1. **Data Processing:** User profiles are converted into feature vectors representing riding preferences
2. **Graph Construction:** A graph network is created where riders are nodes and similarities form edges
3. **Neural Network Learning:** A GNN learns optimal embeddings for users based on their attributes
4. **Match Generation:** Similar riders are identified through embedding space proximity
5. **Visualization:** Network graphs and embedding spaces showcase community clusters

## 👤 User Attributes

The platform analyzes several key attributes:

- **Riding Experience:** Years of riding experience
- **EV Model:** Specific RapteeHV model (Standard, City Edition, Pro, Touring, Adventure)
- **Location:** City/region for proximity matching
- **Riding Style:** Preferences like commuting, sport riding, touring, etc.
- **Interests:** Personal interests beyond riding
- **Performance Mode:** Preferred motorcycle performance setting
- **Group Memberships:** Existing rider group affiliations
- **Favorite Routes:** Preferred riding paths
- **Usage Statistics:** Distance traveled, energy savings, carbon offset

## 📥 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ELEVATE.git
cd ELEVATE

# Install dependencies
pip install torch torch-geometric plotly networkx numpy pandas

# Run the application
python main.py
```

## 🚀 Usage Examples

### Finding Matches for a Specific User

```python
# Get matches for user5 (Vikram Singh)
json_result = get_matches_json(
    "Vikram Singh", 
    "user5", 
    users, 
    user_embeddings, 
    similarity_engine, 
    max_recommendations=5
)
```

### Visualizing the Network

```python
# Generate network visualization
visualize_user_graph(G, user_to_idx, "RapteeHV Riders Network")

# Show user clusters based on embeddings
visualize_user_clusters(user_embeddings, users, "Rider Community Segments")

# Visualize matches for a specific user
visualize_match_results(user, matches, match_explanations)
```

## 🔮 Future Developments

- **Real-time Geolocation:** Incorporate live rider positions for immediate meetup suggestions
- **Ride Planning:** Collaborative route creation and planning features
- **Event Recommendations:** Suggest gatherings based on rider clusters and interests
- **Mobile App Integration:** Native mobile interfaces for Android and iOS
- **Charging Station Data:** Incorporate charging network information for group ride planning

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.