import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_similarity():
    data = pd.read_csv('data/latest_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def recommend_movies(movie_name):
    movie_name = movie_name.lower()
    data, similarity = create_similarity()
    if movie_name not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        movie_index = data.loc[data['movie_title']==movie_name].index[0]
        similarity_scores = list(enumerate(similarity[movie_index]))
        sorted_similarity_scores = sorted(similarity_scores, key = lambda x:x[1] ,reverse=True)
        recommended_similarity_scores = sorted_similarity_scores[1:11] # excluding first item since it is the requested movie itself
        recommended_movie_names = []
        for i in range(len(recommended_similarity_scores)):
            recommended_movie_index = recommended_similarity_scores[i][0]
            recommended_movie_names.append(data['movie_title'][recommended_movie_index])
        return recommended_movie_names
    
movie_name = 'John Wick: Chapter 4'
recommended_movies = recommend_movies(movie_name)
print(recommended_movies)