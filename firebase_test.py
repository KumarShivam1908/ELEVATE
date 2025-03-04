import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Initialize Firebase
cred = credentials.Certificate(r"C:\Users\YasirAhmd\OneDrive\Desktop\codevolt\client_secret_295333690624-gv5nbl9hajegl258hm7v7uoap6nosvpd.apps.googleusercontent.com.json")

# Check if the app is already initialized to avoid duplicate initialization error
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Get Firestore database reference
db = firestore.client()

# Fetch user data
def get_users():
    users_ref = db.collection("users")
    docs = users_ref.stream()

    for doc in docs:
        print(f"User ID: {doc.id}")
        print(f"User Data: {doc.to_dict()}\n")

# Call function to fetch and display users
get_users()
