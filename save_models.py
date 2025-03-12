import pandas as pd
import numpy as np
import joblib
import pickle
import os

print("Current versions:")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")

# Load your original pickle files
try:
    with open('model/movies_list.pkl', 'rb') as f:
        movies = pickle.load(f)
    with open('model/similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
    
    print("Original files loaded successfully")
    print(f"Movies shape: {movies.shape}")
    print(f"Similarity shape: {similarity.shape}")
    
    # Save in multiple formats for compatibility
    # Save as joblib
    joblib.dump(movies, 'model/movies_list_new.pkl')
    joblib.dump(similarity, 'model/similarity_new.pkl')
    
    # Save DataFrame using pandas
    movies.to_pickle('model/movies_list_pd.pkl')
    
    # Save similarity matrix using numpy
    np.save('model/similarity_np.npy', similarity)
    
    print("Files saved in new formats successfully")
    
except Exception as e:
    print(f"Error: {str(e)}") 