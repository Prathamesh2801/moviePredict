# Movie Recommendation System

A Flask-based web application that recommends movies based on user selection. The system uses machine learning to provide personalized movie recommendations along with movie posters from TMDB API.

## Features

- Movie recommendation based on content similarity
- Movie poster display using TMDB API
- User-friendly web interface
- About and Contact pages

## Installation

1. Clone the repository
```bash
git clone <your-repo-url>
cd MovieRecommendersystem
```

2. Create a virtual environment
```bash
python -m venv moviesenv
source moviesenv/bin/activate  # On Windows: moviesenv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
python app.py
```

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
- `static/`: Static files (CSS, JS, images)
- `model/`: Trained ML models and data files
- `Dataset/`: Original dataset used for training

## Technologies Used

- Flask
- Python
- Scikit-learn
- TMDB API
- HTML/CSS
- Bootstrap 