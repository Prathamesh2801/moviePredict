import os
from flask import Flask, request, render_template
import pickle
import requests
import pandas as pd
from patsy import dmatrices

# Load models
try:
    movies = pickle.load(open('model/movies_list.pkl', 'rb'))
    similarity = pickle.load(open('model/similarity.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model files: {e}")
    raise

def fetch_poster(movie_id):
    try:
        url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
        data = requests.get(url)
        data = data.json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    except Exception as e:
        print(f"Error fetching poster: {e}")
        return None

def recommend(movie):
    try:
        index = movies[movies['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_movies_name = []
        recommended_movies_poster = []
        for i in distances[1:6]:
            movies_id = movies.iloc[i[0]].movie_id
            poster = fetch_poster(movies_id)
            if poster:
                recommended_movies_poster.append(poster)
                recommended_movies_name.append(movies.iloc[i[0]].title)

        return recommended_movies_name, recommended_movies_poster
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return [], []

app = Flask(__name__)
# Set environment-specific configurations
app.config['ENV'] = os.environ.get('FLASK_ENV', 'production')
app.config['DEBUG'] = False if app.config['ENV'] == 'production' else True

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    movie_list = movies['title'].values
    status = False
    error = None

    if request.method == 'POST':
        try:
            if request.form:
                movies_name = request.form['movies']
                if not movies_name:
                    raise ValueError("No movie selected")
                
                recommended_movies_name, recommended_movies_poster = recommend(movies_name)
                if not recommended_movies_name:
                    raise ValueError("No recommendations found")
                
                status = True
                return render_template(
                    "prediction.html",
                    movies_name=recommended_movies_name,
                    poster=recommended_movies_poster,
                    movie_list=movie_list,
                    status=status
                )

        except Exception as e:
            error = str(e)
            print(f"Error in prediction: {e}")

    return render_template(
        "prediction.html",
        movie_list=movie_list,
        status=status,
        error=error
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
