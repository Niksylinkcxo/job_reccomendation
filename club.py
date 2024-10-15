from flask import Flask, request, jsonify
import pandas as pd
import logging
import random
from pymongo import MongoClient
import google.generativeai as genai

app = Flask(__name__)

# MongoDB connection setup for a specific database
def connect_mongodb(uri, database_name):
    client = MongoClient(uri)
    return client[database_name]

# Load club data from MongoDB collection in the club database
def load_club_data_from_mongo(club_db):
    collection = club_db["clubs"]  # Replace with your actual club collection name
    club_data = list(collection.find())
    print(f"Loaded {len(club_data)} clubs from MongoDB")
    if len(club_data) == 0:
        print("No club data found in MongoDB.")
    return pd.DataFrame(club_data)

# Load user details data from MongoDB collection in the user database
def load_user_data_from_mongo(user_db):
    collection = user_db["user"]  # Replace with your actual user collection name
    user_data = list(collection.find())
    print(f"Loaded {len(user_data)} users from MongoDB")
    return pd.DataFrame(user_data)

# Set up Google Generative AI with multiple API keys
api_list = [
    "AIzaSyBh8aCHarl5Dkqkh9yB-OBHgrmudp8WAXA",
    "AIzaSyA9tAxLYzBX9Xg1slh9EO9SBEW-VW5OuJo"
]
api_key = random.choice(api_list)
print(f"Using API Key: {api_key}")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to generate a response using the Gemini model
def get_gemini_response(user_text, club_strings, prompt):
    responses = []
    for club_string in club_strings:
        try:
            response = model.generate_content([user_text, club_string, prompt], generation_config=genai.GenerationConfig(
                temperature=0.4
            ))
            responses.append(response.text)
        except Exception as e:
            logging.error(f"Error generating Gemini response: {e}")
            responses.append("Error generating response.")
    return responses

def get_club_recommendations(designation, organization, industries=None, location=None, interest=None, board=None):
    user_text = f"{designation} {organization}"
    
    if industries:
        user_text += f" {industries}"
    if location:
        user_text += f" {location}"
    if interest:
        user_text += f" {interest}"
    if board:
        user_text += f" {board}"
    
    logging.info(f"User Input: {user_text}")
    
    # Check if club list DataFrame is loaded properly
    print(f"Club List Columns: {club_list.columns}")
    print(f"Club List Preview before filtering: {club_list.head()}")
    
    # If club_list is empty, stop further processing
    if club_list.empty:
        print("Club list is empty. Cannot proceed with recommendations.")
        return []

    # Generate club strings for the Gemini model
    club_strings = club_list.apply(lambda row: f"ID: {row['_id']} Title: {row['title']} Details: {row['details']}", axis=1).tolist()
    print(f"Generated club strings for model input: {club_strings[:15]}")  # Preview only first 5 clubs

    # Set up a prompt for the Gemini model
    prompt = (
        "Based on the user's profile and interests, identify and recommend the most relevant clubs from the provided list. "
        "Rank the clubs in order of relevance."
    )
    
    responses = get_gemini_response(user_text, club_strings, prompt)
    
    scores_with_indices = []
    for idx, responses in enumerate(responses):
        try:
            relevance_score = random.uniform(0, 100)  # Placeholder for a real scoring mechanism
            scores_with_indices.append((relevance_score, idx))
            print(f"Generated relevance score {relevance_score:.2f} for club at index {idx}")
        except Exception as e:
            logging.error(f"Error processing Gemini response: {e}")
            continue
    
    scores_with_indices.sort(reverse=True, key=lambda x: x[0])
    print(f"Sorted club scores: {scores_with_indices}")

    all_indices = [idx for _, idx in scores_with_indices]
    recommended_clubs = club_list.iloc[all_indices].copy()
    
    recommended_clubs['similarity_score'] = [f"{score:.2f}" for score, _ in scores_with_indices]
    print(f"Recommended clubs with similarity scores: {recommended_clubs.head()}")

    # Convert _id to id and format output with all necessary fields
    recommended_clubs['id'] = recommended_clubs['_id']

    return recommended_clubs[[ 'id', 'authorId', 'authorType', 'bannerImage', 'title', 'details', 
    'isPrivate', 'fee', 'feecurrency', 'noOfApplicants', 'industry', 
    'function', 'category', 'picUrl', 'createdAt', 'updatedAt', 'oldId','similarity_score',]].reset_index(drop=True).to_dict(orient='records')

@app.route('/v1/get_club_recommendations', methods=['GET'])
def recommendations():
    # Get query parameters
    designation = request.args.get('designation')
    organization = request.args.get('organization')
    industry = request.args.get('industry')
    location = request.args.get('location')
    interest = request.args.get('interest')
    board = request.args.get('board')
    page = int(request.args.get('page', 1))  # Default to page 1 if not provided
    limit = int(request.args.get('limit', 5))  # Default to limit 20 if not provided

    # Validate required parameters
    if not designation or not organization:
        return jsonify({'error': 'Designation and Organization are required'}), 400

    # Call the club recommendation function
    recommendations = get_club_recommendations(designation, organization, industry, location, interest, board)

    # Implement pagination
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_recommendations = recommendations[start_index:end_index]

    # Prepare the response
    response = {
        'page': page,
        'limit': limit,
        'total_recommendations': len(recommendations),
        'total_pages': (len(recommendations) + limit - 1) // limit,  # Calculate total pages
        'recommendations': paginated_recommendations
    }

    return jsonify(response)

if __name__ == '__main__':
    # You could initialize the MongoDB connections and data here.
    mongo_uri = "mongodb+srv://linkcxoadmin:PxDpyQH3p1ie1nq@prod-cluster.rtezr.mongodb.net/"  # Replace with your MongoDB URI
    club_database_name = "api-club-service"  # Replace with your club database name
    user_database_name = "api-user-service"  # Replace with your user database name

    # Connect to both MongoDB databases
    club_db = connect_mongodb(mongo_uri, club_database_name)
    user_db = connect_mongodb(mongo_uri, user_database_name)

    # Load club and user data from respective databases
    club_list = load_club_data_from_mongo(club_db)
    user_details = load_user_data_from_mongo(user_db)

    app.run(debug=False)
