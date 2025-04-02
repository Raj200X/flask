import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from flask import Flask, render_template, request, jsonify, abort, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import json
import os
import signal
import sys
from textblob import TextBlob
import requests
from datetime import datetime
import time
from dotenv import load_dotenv
from models import db, User, Favorite, UserPreference
from werkzeug.security import generate_password_hash, check_password_hash
import re
import google.generativeai as genai
from flask_cors import CORS
import sqlite3

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movies.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)  # Enable CORS for all routes

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# TMDB API Configuration
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
if not TMDB_API_KEY:
    print("Error: TMDB_API_KEY not found in environment variables")
    sys.exit(1)

TMDB_BASE_URL = 'https://api.themoviedb.org/3'
TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'

# OMDB API configuration
OMDB_API_KEY = os.getenv('OMDB_API_KEY', 'your-omdb-api-key')
OMDB_BASE_URL = 'http://www.omdbapi.com/'

# Initialize Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

print("Initializing Gemini with API key:", GOOGLE_API_KEY[:10] + "...")
genai.configure(api_key=GOOGLE_API_KEY)

# Set up the model with the correct version and configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the model with the correct version
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",  # Updated to use the latest model version
    generation_config=generation_config,
    safety_settings=safety_settings
)
print("Gemini model initialized successfully")

# Create a chat context that focuses on movies
MOVIE_CONTEXT = """You are a friendly and knowledgeable movie recommendation assistant. Your purpose is to help users discover movies they might enjoy and provide information about films. You can:

1. Recommend movies based on user preferences
2. Provide information about specific movies
3. Suggest similar movies
4. Discuss movie genres, directors, actors, and themes
5. Share interesting movie facts and trivia

Important rules:
- Only discuss topics related to movies and cinema
- If asked about non-movie topics, politely redirect the conversation back to movies
- Keep responses friendly but professional
- Avoid sharing inappropriate content
- Don't recommend movies that aren't suitable for the user's age group
- If unsure about a movie detail, acknowledge the uncertainty

Please start by asking how you can help with movie recommendations today."""

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def search_tmdb_movies(query):
    """Search for movies using TMDB API with enhanced results."""
    try:
        url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'language': 'en-US',
            'include_adult': False
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        movies = response.json()['results']

        # Get additional details for each movie
        enhanced_movies = []
        for movie in movies:
            # Get movie details including credits
            details_url = f"{TMDB_BASE_URL}/movie/{movie['id']}"
            details_params = {
                'api_key': TMDB_API_KEY,
                'language': 'en-US',
                'append_to_response': 'credits'
            }
            details_response = requests.get(details_url, params=details_params)
            details_response.raise_for_status()
            movie_details = details_response.json()

            # Get cast members
            cast = []
            if 'credits' in movie_details and 'cast' in movie_details['credits']:
                cast = [actor['name'] for actor in movie_details['credits']['cast'][:5]]

            enhanced_movies.append({
                'id': movie['id'],
                'title': movie['title'],
                'release_date': movie['release_date'],
                'vote_average': movie['vote_average'],
                'popularity': movie['popularity'],
                'poster_path': movie['poster_path'],
                'cast': cast,
                'overview': movie['overview']
            })
            time.sleep(0.25)  # Rate limiting

        return enhanced_movies
    except Exception as e:
        print(f"Error searching movies: {str(e)}")
        return []

def get_tmdb_movies(page=1):
    """Fetch popular movies from TMDB API."""
    try:
        # Get top-rated movies
        url = f"{TMDB_BASE_URL}/movie/top_rated"
        params = {
            'api_key': TMDB_API_KEY,
            'language': 'en-US',
            'page': page
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        top_rated = response.json()['results']

        # Get popular movies
        url = f"{TMDB_BASE_URL}/movie/popular"
        response = requests.get(url, params=params)
        response.raise_for_status()
        popular = response.json()['results']

        # Combine and sort by rating
        all_movies = top_rated + popular
        all_movies.sort(key=lambda x: x['vote_average'], reverse=True)
        
        # Remove duplicates based on movie ID
        seen_ids = set()
        unique_movies = []
        for movie in all_movies:
            if movie['id'] not in seen_ids:
                seen_ids.add(movie['id'])
                unique_movies.append(movie)

        return unique_movies[:20]  # Return top 20 movies
    except Exception as e:
        print(f"Error fetching movies from TMDB: {str(e)}")
        return []

def get_movie_details(movie_id):
    """Fetch detailed movie information from both TMDB and OMDB APIs"""
    # TMDB API call
    tmdb_url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    tmdb_params = {
        'api_key': TMDB_API_KEY,
        'append_to_response': 'credits,similar'
    }
    tmdb_response = requests.get(tmdb_url, params=tmdb_params)
    tmdb_data = tmdb_response.json()

    if tmdb_response.status_code != 200:
        return None

    # OMDB API call
    omdb_url = OMDB_BASE_URL
    omdb_params = {
        'apikey': OMDB_API_KEY,
        'i': f"tt{tmdb_data.get('imdb_id', '')}",
        'plot': 'full'
    }
    omdb_response = requests.get(omdb_url, params=omdb_params)
    omdb_data = omdb_response.json()

    # Combine data from both APIs
    movie_details = {
        'id': movie_id,
        'title': tmdb_data.get('title'),
        'description': tmdb_data.get('overview'),
        'year': tmdb_data.get('release_date', '')[:4],
        'rating': tmdb_data.get('vote_average', 0),
        'genres': [genre['name'] for genre in tmdb_data.get('genres', [])],
        'runtime': tmdb_data.get('runtime', 0),
        'poster_url': f"{TMDB_IMAGE_BASE_URL}{tmdb_data.get('poster_path')}" if tmdb_data.get('poster_path') else None,
        'director': next((crew['name'] for crew in tmdb_data.get('credits', {}).get('crew', []) 
                         if crew['job'] == 'Director'), 'Unknown'),
        'cast': [{'name': cast['name'], 'character': cast['character']} 
                for cast in tmdb_data.get('credits', {}).get('cast', [])[:5]],
        'awards': omdb_data.get('Awards', 'N/A'),
        'box_office': omdb_data.get('BoxOffice', 'N/A'),
        'production': omdb_data.get('Production', 'N/A'),
        'website': omdb_data.get('Website', 'N/A'),
        'similar_movies': tmdb_data.get('similar', {}).get('results', [])[:5]
    }
    return movie_details

def analyze_sentiment(text):
    """Analyze the sentiment of a text using TextBlob."""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def create_movie_features(movies):
    """Create feature vectors for movies based on multiple features."""
    try:
        features = []
        for movie in movies:
            # Combine all text features
            text_features = ' '.join(movie['genres']) + ' ' + movie['description']
            
            # Calculate average sentiment from reviews
            sentiments = [analyze_sentiment(review) for review in movie['reviews']]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            # Create numerical features
            numerical_features = [
                movie['rating'],
                avg_sentiment,
                movie['year'],
                movie['budget'] / 1000000,  # Convert to millions
                movie['revenue'] / 1000000,  # Convert to millions
                movie['runtime']
            ]
            
            features.append({
                'text': text_features,
                'numerical': numerical_features
            })
        return features
    except Exception as e:
        print(f"Error creating movie features: {str(e)}")
        return []

def get_recommendations(movie_id, n_recommendations=3):
    """Get movie recommendations using hybrid content-based and collaborative filtering."""
    try:
        # Get all movies from TMDB
        movies = []
        for page in range(1, 3):  # Get first 2 pages of popular movies
            tmdb_movies = get_tmdb_movies(page)
            for tmdb_movie in tmdb_movies:
                movie_details = get_movie_details(tmdb_movie['id'])
                if movie_details:
                    movies.append(movie_details)
            time.sleep(1)  # Rate limiting

        features = create_movie_features(movies)
        
        if not features:
            return []
        
        # Create TF-IDF vectors for text features
        text_vectorizer = TfidfVectorizer(stop_words='english')
        text_matrix = text_vectorizer.fit_transform([f['text'] for f in features])
        
        # Create numerical feature matrix
        numerical_matrix = np.array([f['numerical'] for f in features])
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_matrix_scaled = scaler.fit_transform(numerical_matrix)
        
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=3)
        numerical_matrix_pca = pca.fit_transform(numerical_matrix_scaled)
        
        # Combine text and numerical features
        combined_matrix = np.hstack([
            text_matrix.toarray(),
            numerical_matrix_pca
        ])
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(combined_matrix)
        
        # Get the index of the movie
        movie_idx = next(i for i, m in enumerate(movies) if m['id'] == movie_id)
        
        # Get similarity scores for the movie
        sim_scores = list(enumerate(cosine_sim[movie_idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return recommended movies with similarity scores
        recommendations = []
        for idx in movie_indices:
            movie = movies[idx]
            similarity = cosine_sim[movie_idx][idx]
            recommendations.append({
                **movie,
                'similarity_score': round(similarity * 100, 2),  # Convert to percentage
                'poster_url': f"{TMDB_IMAGE_BASE_URL}{movie['poster_path']}" if movie['poster_path'] else None
            })
        
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        return []

def analyze_movie_sentiment(movie_details):
    """Analyze movie sentiment using TextBlob"""
    text = f"{movie_details['description']} {' '.join(movie_details['genres'])}"
    analysis = TextBlob(text)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }

def get_similar_movies(movie_id, n_recommendations=5):
    """Get similar movies based on genre, sentiment, and rating"""
    try:
        # Get the target movie details
        target_movie = get_movie_details(movie_id)
        if not target_movie:
            return []

        # Get target movie sentiment
        target_sentiment = analyze_movie_sentiment(target_movie)
        target_genres = set(target_movie['genres'])
        target_rating = target_movie['rating']

        # Get recommendations from TMDB
        tmdb_url = f"{TMDB_BASE_URL}/movie/{movie_id}/recommendations"
        tmdb_params = {
            'api_key': TMDB_API_KEY,
            'language': 'en-US',
            'page': 1
        }
        response = requests.get(tmdb_url, params=tmdb_params)
        recommendations = response.json().get('results', [])

        # Process and score each recommendation
        scored_movies = []
        for movie in recommendations:
            # Get detailed movie information
            movie_details = get_movie_details(movie['id'])
            if not movie_details:
                continue

            # Calculate similarity score
            movie_genres = set(movie_details['genres'])
            genre_similarity = len(target_genres.intersection(movie_genres)) / len(target_genres.union(movie_genres))
            
            movie_sentiment = analyze_movie_sentiment(movie_details)
            sentiment_similarity = 1 - abs(target_sentiment['polarity'] - movie_sentiment['polarity'])
            
            rating_similarity = 1 - abs(target_rating - movie_details['rating']) / 10

            # Weighted average of similarity scores
            similarity_score = (
                0.4 * genre_similarity +
                0.3 * sentiment_similarity +
                0.3 * rating_similarity
            )

            scored_movies.append({
                'id': movie_details['id'],
                'title': movie_details['title'],
                'poster_url': movie_details['poster_url'],
                'rating': movie_details['rating'],
                'similarity_score': similarity_score
            })

        # Sort by similarity score and return top N
        scored_movies.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_movies[:n_recommendations]

    except Exception as e:
        print(f"Error getting similar movies: {str(e)}")
        return []

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()

        login_user(user)
        return redirect(url_for('home'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('home'))

        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    return redirect(url_for('home'))

@app.route('/profile')
@login_required
def profile():
    """Display user profile and preferences"""
    return render_template('profile.html')

@app.route('/favorites')
@login_required
def favorites():
    user_favorites = Favorite.query.filter_by(user_id=current_user.id).all()
    return render_template('favorites.html', favorites=user_favorites)

@app.route('/favorites/<int:movie_id>', methods=['POST'])
@login_required
def toggle_favorite(movie_id):
    """Toggle favorite status for a movie"""
    try:
        movie_details = get_movie_details(movie_id)
        if not movie_details:
            return jsonify({'status': 'error', 'message': 'Movie not found'})

        favorite = Favorite.query.filter_by(
            user_id=current_user.id,
            movie_id=movie_id
        ).first()

        if favorite:
            db.session.delete(favorite)
            status = 'removed'
        else:
            new_favorite = Favorite(
                user_id=current_user.id,
                movie_id=movie_id,
                movie_title=movie_details['title'],
                poster_url=movie_details['poster_url']
            )
            db.session.add(new_favorite)
            status = 'added'

        db.session.commit()
        return jsonify({'status': status})

    except Exception as e:
        print(f"Error toggling favorite: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/search')
def search():
    """Search for movies"""
    query = request.args.get('q', '')
    if not query:
        return redirect(url_for('home'))

    try:
        # Search TMDB API
        tmdb_url = f"{TMDB_BASE_URL}/search/movie"
        tmdb_params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'language': 'en-US',
            'page': 1
        }
        response = requests.get(tmdb_url, params=tmdb_params)
        search_results = response.json().get('results', [])

        # Process search results
        movies = []
        for movie in search_results:
            movie_details = get_movie_details(movie['id'])
            if movie_details:
                movies.append(movie_details)

        return render_template('search_results.html',
                             movies=movies,
                             query=query,
                             user_favorites=[f.movie_id for f in current_user.favorites] if current_user.is_authenticated else [])

    except Exception as e:
        print(f"Error in search route: {str(e)}")
        return render_template('search_results.html',
                             movies=[],
                             query=query,
                             user_favorites=[])

@app.route('/')
def home():
    """Display popular movies and personalized recommendations"""
    try:
        # Fetch popular movies from TMDB
        tmdb_url = f"{TMDB_BASE_URL}/movie/popular"
        tmdb_params = {
            'api_key': TMDB_API_KEY,
            'language': 'en-US',
            'page': 1
        }
        response = requests.get(tmdb_url, params=tmdb_params)
        popular_movies = response.json().get('results', [])

        # Process popular movies
        movies = []
        for movie in popular_movies[:12]:  # Limit to 12 movies
            movie_details = get_movie_details(movie['id'])
            if movie_details:
                movies.append(movie_details)

        # Get personalized recommendations if user is logged in
        personalized_recommendations = []
        if current_user.is_authenticated:
            user_prefs = current_user.preferences
            if user_prefs:
                # Get recommendations based on user preferences
                favorite_genres = json.loads(user_prefs.favorite_genres)
                min_rating = user_prefs.min_rating
                
                # Fetch movies matching user preferences
                tmdb_url = f"{TMDB_BASE_URL}/discover/movie"
                tmdb_params = {
                    'api_key': TMDB_API_KEY,
                    'with_genres': ','.join(favorite_genres),
                    'vote_average.gte': min_rating,
                    'sort_by': 'popularity.desc',
                    'page': 1
                }
                response = requests.get(tmdb_url, params=tmdb_params)
                recommended_movies = response.json().get('results', [])
                
                # Process recommended movies
                for movie in recommended_movies[:6]:  # Limit to 6 recommendations
                    movie_details = get_movie_details(movie['id'])
                    if movie_details:
                        personalized_recommendations.append(movie_details)

        return render_template('index.html', 
                             movies=movies,
                             personalized_recommendations=personalized_recommendations,
                             user_favorites=[f.movie_id for f in current_user.favorites] if current_user.is_authenticated else [])

    except Exception as e:
        print(f"Error in home route: {str(e)}")
        return render_template('index.html', 
                             movies=[],
                             personalized_recommendations=[],
                             user_favorites=[])

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    """Display detailed information about a specific movie"""
    try:
        movie_details = get_movie_details(movie_id)
        if not movie_details:
            return render_template('404.html'), 404

        # Get user favorites if logged in
        user_favorites = []
        if current_user.is_authenticated:
            user_favorites = [f.movie_id for f in current_user.favorites]

        return render_template('movie_details.html',
                             movie=movie_details,
                             user_favorites=user_favorites)

    except Exception as e:
        print(f"Error in movie_details route: {str(e)}")
        return render_template('404.html'), 404

@app.route('/recommendations/<int:movie_id>')
def get_recommendations(movie_id):
    """API endpoint for getting similar movies"""
    try:
        similar_movies = get_similar_movies(movie_id)
        return jsonify(similar_movies)
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        return jsonify([])

@app.route('/update_preferences', methods=['POST'])
@login_required
def update_preferences():
    """Update user preferences"""
    try:
        preferences = request.json
        user_prefs = current_user.preferences

        if not user_prefs:
            user_prefs = UserPreference(user_id=current_user.id)

        user_prefs.favorite_genres = json.dumps(preferences.get('genres', []))
        user_prefs.favorite_directors = json.dumps(preferences.get('directors', []))
        user_prefs.min_rating = float(preferences.get('min_rating', 7.0))
        user_prefs.preferred_years = json.dumps(preferences.get('years', []))

        if not current_user.preferences:
            db.session.add(user_prefs)
        db.session.commit()

        return jsonify({'status': 'success'})

    except Exception as e:
        print(f"Error updating preferences: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    print('\nShutting down gracefully...')
    sys.exit(0)

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'response': 'Please enter a message.',
                'type': 'error'
            })

        try:
            # Get API key from environment
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")

            # Prepare the API request
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
            
            # Prepare the request body with movie context
            request_body = {
                "contents": [{
                    "parts": [{
                        "text": f"""You are a friendly and knowledgeable movie recommendation assistant. 
                        Your purpose is to help users discover movies they might enjoy and provide information about films.
                        
                        User message: {user_message}"""
                    }]
                }]
            }

            # Make the API request
            response = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=request_body
            )

            # Check if the request was successful
            if not response.ok:
                print(f"API Error Response: {response.text}")
                raise Exception(f"API request failed with status {response.status_code}")

            # Parse the response
            response_data = response.json()
            
            # Extract the generated text
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
                return jsonify({
                    'response': generated_text,
                    'type': 'text'
                })
            else:
                raise Exception("No response content from API")

        except Exception as e:
            print(f"Detailed API Error: {str(e)}")
            print(f"Error type: {type(e)}")
            
            if "API key" in str(e).lower():
                return jsonify({
                    'response': "API key configuration issue. Please check the settings.",
                    'type': 'error'
                })
            elif "rate limit" in str(e).lower():
                return jsonify({
                    'response': "Too many requests. Please try again in a few seconds.",
                    'type': 'error'
                })
            else:
                return jsonify({
                    'response': f"Error processing request: {str(e)}",
                    'type': 'error'
                })

    except Exception as e:
        print(f"General Error: {str(e)}")
        return jsonify({
            'response': 'An error occurred. Please try again.',
            'type': 'error'
        })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True) 