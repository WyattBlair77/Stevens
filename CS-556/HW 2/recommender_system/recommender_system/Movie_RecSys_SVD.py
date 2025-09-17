#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendations HW

# **Name:**  

# **Collaboration Policy:** Homeworks will be done individually: each student must hand in their own answers. Use of partial or entire solutions obtained from others or online is strictly prohibited.

# **Late Policy:** Late submission have a penalty of 2\% for each passing hour. 

# **Submission format:** Successfully complete the Movie Lens recommender as described in this jupyter notebook. Submit a `.py` and an `.ipynb` file for this notebook. You can go to `File -> Download as ->` to download a .py version of the notebook. 
# 
# **Only submit one `.ipynb` file and one `.py` file.** The `.ipynb` file should have answers to all the questions. Do *not* zip any files for submission. 

# **Download the dataset from here:** https://grouplens.org/datasets/movielens/1m/

# In[1]:


# Import all the required libraries
import numpy as np
import pandas as pd


# ## Reading the Data
# Now that we have downloaded the files from the link above and placed them in the same directory as this Jupyter Notebook, we can load each of the tables of data as a CSV into Pandas. Execute the following, provided code.

# In[2]:


# Read the dataset from the two files into ratings_data and movies_data
#NOTE: if you are getting a decode error, add "encoding='ISO-8859-1'" as an additional argument
#      to the read_csv function
column_list_ratings = ["UserID", "MovieID", "Ratings","Timestamp"]
ratings_data  = pd.read_csv('ratings.dat',sep='::',names = column_list_ratings, engine='python')
column_list_movies = ["MovieID","Title","Genres"]
movies_data = pd.read_csv('movies.dat',sep = '::',names = column_list_movies, engine='python', encoding = 'latin-1')
column_list_users = ["UserID","Gender","Age","Occupation","Zixp-code"]
user_data = pd.read_csv("users.dat",sep = "::",names = column_list_users, engine='python')


# `ratings_data`, `movies_data`, `user_data` corresponds to the data loaded from `ratings.dat`, `movies.dat`, and `users.dat` in Pandas.

# ## Data analysis

# We now have all our data in Pandas - however, it's as three separate datasets! To make some more sense out of the data we have, we can use the Pandas `merge` function to combine our component data-frames. Run the following code:

# In[3]:


data=pd.merge(pd.merge(ratings_data,user_data),movies_data)
data


# Next, we can create a pivot table to match the ratings with a given movie title. Using `data.pivot_table`, we can aggregate (using the average/`mean` function) the reviews and find the average rating for each movie. We can save this pivot table into the `mean_ratings` variable. 

# In[4]:


mean_ratings=data.pivot_table('Ratings','Title',aggfunc='mean')
mean_ratings


# Now, we can take the `mean_ratings` and sort it by the value of the rating itself. Using this and the `head` function, we can display the top 15 movies by average rating.

# In[5]:


mean_ratings=data.pivot_table('Ratings',index=["Title"],aggfunc='mean')
top_15_mean_ratings = mean_ratings.sort_values(by = 'Ratings',ascending = False).head(15)
top_15_mean_ratings


# Let's adjust our original `mean_ratings` function to account for the differences in gender between reviews. This will be similar to the same code as before, except now we will provide an additional `columns` parameter which will separate the average ratings for men and women, respectively.

# In[6]:


mean_ratings=data.pivot_table('Ratings',index=["Title"],columns=["Gender"],aggfunc='mean')
mean_ratings


# We can now sort the ratings as before, but instead of by `Rating`, but by the `F` and `M` gendered rating columns. Print the top rated movies by male and female reviews, respectively.

# In[7]:


data=pd.merge(pd.merge(ratings_data,user_data),movies_data)

mean_ratings=data.pivot_table('Ratings',index=["Title"],columns=["Gender"],aggfunc='mean')
top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)
print(top_female_ratings.head(15))

top_male_ratings = mean_ratings.sort_values(by='M', ascending=False)
print(top_male_ratings.head(15))


# In[8]:


mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff')
sorted_by_diff[:10]


# Let's try grouping the data-frame, instead, to see how different titles compare in terms of the number of ratings. Group by `Title` and then take the top 10 items by number of reviews. We can see here the most popularly-reviewed titles.

# In[9]:


ratings_by_title=data.groupby('Title').size()
ratings_by_title.sort_values(ascending=False).head(10)


# Similarly, we can filter our grouped data-frame to get all titles with a certain number of reviews. Filter the dataset to get all movie titles such that the number of reviews is >= 2500.

# ## Question 1

# Create a ratings matrix using Numpy. This matrix allows us to see the ratings for a given movie and user ID. The element at location $[i,j]$ is a rating given by user $i$ for movie $j$. Print the **shape** of the matrix produced.  
# 
# Additionally, choose 3 users that have rated the movie with MovieID "**1377**" (Batman Returns). Print these ratings, they will be used later for comparison.
# 
# 
# **Notes:**
# - Do *not* use `pivot_table`.
# - A ratings matrix is *not* the same as `ratings_data` from above.
# - The ratings of movie with MovieID $i$ are stored in the ($i$-1)th column (index starts from 0)  
# - Not every user has rated every movie. Missing entries should be set to 0 for now.
# - If you're stuck, you might want to look into `np.zeros` and how to use it to create a matrix of the desired shape.
# - Every review lies between 1 and 5, and thus fits within a `uint8` datatype, which you can specify to numpy.

# In[10]:


# Define helpful constants

max_movie_id, min_movie_id = data.MovieID.max(), 1
max_user_id, min_user_id = data.UserID.max(), 1

movie_ids, user_ids = np.arange(min_movie_id, max_movie_id), np.arange(min_user_id, max_user_id)
num_movies, num_users = len(movie_ids)+1, len(user_ids)


# In[11]:


# Define a function which quickly generates a row in the matrix

def get_ratings(user_id):
    
    relevant_data = data[data['UserID'] == user_id]
    ratings = np.zeros(num_movies)
    for movie_id in relevant_data.MovieID.values:
        
        ratings[movie_id - 1] = relevant_data[relevant_data.MovieID == movie_id].Ratings.values[0]
        
    return ratings


# In[12]:


# Create the matrix

ratings_matrix = np.zeros((num_users, num_movies))

for i, user_id in enumerate(user_ids):
    
    print('CONSTRUCTING ROW: %i/%i (%1.2f %%)' % (i, num_users, (i/num_users * 100)), end='\r')
    ratings_matrix[i] = get_ratings(user_id)
    


# In[13]:


# Print the shape

print(ratings_matrix.shape)


# In[14]:


# Store and print ratings for Batman Returns

batman_users_indicies = [9, 12, 17]

batman_rating1 = ratings_matrix[batman_users_indicies[0], 1376]
batman_rating2 = ratings_matrix[batman_users_indicies[1], 1376]
batman_rating3 = ratings_matrix[batman_users_indicies[2], 1376]

batman_ratings = [batman_rating1, batman_rating2, batman_rating3]

for bm_user_id, bm_rating in zip(batman_users_indicies, batman_ratings):
    print('USER ID: %i --> RATING: %i' % (bm_user_id+1, bm_rating))


# ## Question 2

# Normalize the ratings matrix (created in **Question 1**) using Z-score normalization. While we can't use `sklearn`'s `StandardScaler` for this step, we can do the statistical calculations ourselves to normalize the data.
# 
# Before you start:
# - Your first step should be to get the average of every *column* of the ratings matrix (we want an average by title, not by user!).
# - Make sure that the mean is calculated considering only non-zero elements. If there is a movie which is rated only by 10 users, we get its mean rating using (sum of the 10 ratings)/10 and **NOT** (sum of 10 ratings)/(total number of users)
# - All of the missing values in the dataset should be replaced with the average rating for the given movie. This is a complex topic, but for our case replacing empty values with the mean will make it so that the absence of a rating doesn't affect the overall average, and it provides an "expected value" which is useful for computing correlations and recommendations in later steps.
# - In our matrix, 0 represents a missing rating.
# - Next, we want to subtract the average from the original ratings thus allowing us to get a mean of 0 in every *column*. It may be very close but not exactly zero because of the limited precision `float`s allow.
# - Lastly, divide this by the standard deviation of the *column*.
# 
# - Not every MovieID is used, leading to zero columns. This will cause a divide by zero error when normalizing the matrix. Simply replace any NaN values in your normalized matrix with 0.

# In[15]:


# Obtain column-wise averages, using only non-zero ratings in the denominator
movie_averages = np.divide(np.sum(ratings_matrix, axis=0), np.count_nonzero(ratings_matrix, axis=0))


# In[16]:


# Assign missing values to the appropriate movie_average value
for movie_average, col in zip(movie_averages, ratings_matrix.T):
    col[col == 0] = movie_average


# In[17]:


# Subtract the average from the original ratings
ratings_matrix -= np.resize(movie_averages, ratings_matrix.shape)


# In[18]:


# Divide each column by the standard deviation of the column
movie_stds = np.array([col.std() for col in ratings_matrix.T])
ratings_matrix /= np.resize(movie_stds, ratings_matrix.shape)


# In[19]:


# Replace any NaN values with 0
ratings_matrix[np.isnan(ratings_matrix)] = 0


# In[20]:


batman_rating1 = ratings_matrix[batman_users_indicies[0], 1376]
batman_rating2 = ratings_matrix[batman_users_indicies[1], 1376]
batman_rating3 = ratings_matrix[batman_users_indicies[2], 1376]

batman_ratings = [batman_rating1, batman_rating2, batman_rating3]

for bm_user_id, bm_rating in zip(batman_users_indicies, batman_ratings):
    print('USER ID: %i --> RATING: %i' % (bm_user_id+1, bm_rating))


# ## Question 3

# We're now going to perform Singular Value Decomposition (SVD) on the normalized ratings matrix from the previous question. Perform the process using numpy, and along the way print the shapes of the $U$, $S$, and $V$ matrices you calculated.

# In[21]:


# Compute the SVD of the normalised matrix
# def perform_SVD(A):
    
#     m, n = A.shape
    
#     # Calculate both inner and outer products of A
#     print('STEP 1 --> ', end='')
#     outer_product = A @ A.T
#     inner_product = A.T @ A
#     print('DONE')
    
#     # Determine eigenvalues and unit-eigenvectors for each product, since we'll need both
#     print('STEP 2 --> ', end='')
#     outer_eigenvalues, outer_eigenvectors = np.linalg.eig(outer_product)
#     inner_eigenvalues, inner_eigenvectors = np.linalg.eig(inner_product)
#     print('DONE')

#     # S is a (m x n) matrix with the singular values on the diagonal
#     print('STEP 3 --> ', end='')
#     singular_values = np.sqrt(outer_eigenvalues)
#     S = np.zeros((m, n))
#     np.fill_diagonal(S, singular_values)
#     print('DONE')

#     # V is a (n x n) matrix with the eigenvectors of the inner product on each row
#     print('STEP 4 --> ', end='')
#     VT = np.column_stack(inner_eigenvectors).T
#     print('DONE')

#     # U is a (m x m) matrix which can be solved for once we have calculate S and V
#     print('STEP 5 --> ', end='')
#     U = np.column_stack(outer_eigenvectors)
#     print('DONE')

#     print('=' * 50)
#     print('SHAPE S: ', S.shape, '| EXPECTED: ', (m , n))
#     print('SHAPE U: ', U.shape, '| EXPECTED: ', (m, m))
#     print('SHAPE V: ', VT.shape, '| EXPECTED: ', (n, n))

#     return U, S, VT


# In[22]:


# Print the shapes
# U, S, VT = perform_SVD(ratings_matrix)


# In[23]:


m, n = ratings_matrix.shape

U, S, VT = np.linalg.svd(ratings_matrix)

print('=' * 50)
print('SHAPE S: ', S.shape, '| EXPECTED: ', (n))
print('SHAPE U: ', U.shape, '| EXPECTED: ', (m, m))
print('SHAPE V: ', VT.shape, '| EXPECTED: ', (n, n))


# ## Question 4

# Reconstruct four rank-k rating matrix $R_k$, where $R_k = U_kS_kV_k^T$ for k = [100, 1000, 2000, 3000]. Using each of $R_k$ make predictions for the 3 users selected in Question 1, for the movie with ID 1377 (Batman Returns). Compare the original ratings with the predicted ratings.

# In[24]:


def reconstruct(rank):
    
    m, n = U.shape[0], VT.shape[1]
    
    U_clipped = U[:, :rank]
    S_clipped = np.diag(S[:rank])
    VT_clipped = VT[:rank, :]
    
    print('=' * 50)
    print('k = %i' % rank)
    print('SHAPE S: ', S_clipped.shape, '| EXPECTED: ', (rank , rank))
    print('SHAPE U: ', U_clipped.shape, '| EXPECTED: ', (m, rank))
    print('SHAPE V: ', VT_clipped.shape, '| EXPECTED: ', (rank, n))
    
    R_k = U_clipped @ S_clipped @ VT_clipped
    return R_k

R100 = reconstruct(100)
R1000 = reconstruct(1000)
R2000 = reconstruct(2000)
R3000 = reconstruct(3000)


# In[25]:


# original
batman_rating1 = ratings_matrix[batman_users_indicies[0], 1376]
batman_rating2 = ratings_matrix[batman_users_indicies[1], 1376]
batman_rating3 = ratings_matrix[batman_users_indicies[2], 1376]

batman_ratings = [batman_rating1, batman_rating2, batman_rating3]

print('original:')
for bm_user_id, bm_rating in zip(batman_users_indicies, batman_ratings):
    print('USER ID: %i --> RATING: %i' % (bm_user_id+1, bm_rating))
print('=' * 50)
    
# predictions
for R, k in zip([R100, R1000, R2000, R3000], [100, 1000, 2000, 3000]):
    
    batman_rating1 = R[batman_users_indicies[0], 1376]
    batman_rating2 = R[batman_users_indicies[1], 1376]
    batman_rating3 = R[batman_users_indicies[2], 1376]

    batman_ratings = [batman_rating1, batman_rating2, batman_rating3]
    
    print('k = %i' % k)
    for bm_user_id, bm_rating in zip(batman_users_indicies, batman_ratings):
        print('USER ID: %i --> RATING: %i' % (bm_user_id+1, bm_rating))
    print('=' * 50)


# In[ ]:





# ## Question 5

# ### Cosine Similarity
# Cosine similarity is a metric used to measure how similar two vectors are. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. Cosine similarity is high if the angle between two vectors is 0, and the output value ranges within $cosine(x,y) \in [0,1]$. $0$ means there is no similarity (perpendicular), where $1$ (parallel) means that both the items are 100% similar.
# 
# $$ cosine(x,y) = \frac{x^T y}{||x|| ||y||}  $$

# **Based on the reconstruction rank-1000 rating matrix $R_{1000}$ and the cosine similarity,** sort the movies which are most similar. You will have a function `top_movie_similarity` which sorts data by its similarity to a movie with ID `movie_id` and returns the top $n$ items, and a second function `print_similar_movies` which prints the titles of said similar movies. Return the top 5 movies for the movie with ID `1377` (*Batman Returns*)
# 
# Note: While finding the cosine similarity, there are a few empty columns which will have a magnitude of **zero** resulting in NaN values. These should be replaced by 0, otherwise these columns will show most similarity with the given movie. 

# In[26]:


# Sort the movies based on cosine similarity
def cosine_similarity(x, y):
    
    numerator = x.T @ y
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    
    return abs(numerator / denominator)
    
def top_movie_similarity(data, movie_id, top_n=5):
    # Movie id starts from 1
    movie_id -= 1
    
    # initialize closest movie list
    closest_movies = [(None, -np.inf)] * top_n
        
    # movie-vector
    x = data[:, movie_id]
    
    for movie, y in enumerate(data.T):
        
        if np.linalg.norm(y) != 0: similarity = cosine_similarity(x, y)
        else: similarity = 0
        
        if movie != movie_id:
        
            try:
                best_index = next(ind for ind, val in enumerate(closest_movies) if val[1] < similarity)

                # if there is an value in the list which is less than this similarity, then add
                # the movie ID and the similarity to the list and remove the final item in the list
                closest_movies.insert(best_index, (movie, similarity))
                closest_movies.pop(-1)
            except StopIteration:
                # if this similarity is not in the top_n closest_movies just move on
                pass

    return closest_movies

def print_similar_movies(movie_titles, top_indices):
    print('Most Similar movies: ')
    for rank, (movie_index, similarity) in enumerate(top_indices):
        print('%i) %s -- Similarity: %1.2f%%' % (rank+1, movie_titles[movie_index+1], similarity * 100))

# Print the top 5 movies for Batman Returns
movie_id = 1377


# In[27]:


def get_movie_titles():
    
    movie_titles = {}
    for ind, (movie_id, title) in data[['MovieID', 'Title']].iterrows():
        if movie_id not in movie_titles.keys():
            movie_titles.update({movie_id: title})
    
    return movie_titles
    


# In[28]:


closest_movies = top_movie_similarity(R1000, movie_id)
movie_titles = get_movie_titles()
print_similar_movies(movie_titles, closest_movies)


# ## Question 6

# ### Movie Recommendations
# Using the same process from Question 5, write `top_user_similarity` which sorts data by its similarity to a user with ID `user_id` and returns the top result. Then find the MovieIDs of the movies that this similar user has rated most highly, but that `user_id` has not yet seen. Find at least 5 movie recommendations for the user with ID `5954` and print their titles.
# 
# Hint: To check your results, find the genres of the movies that the user likes and compare with the genres of the recommended movies.

# In[29]:


#Sort users based on cosine similarity
def find_unseen_movies(user_id):
    
    movies = ratings_matrix[user_id, :]
    unseen_movie_indices = np.argwhere(movies == 0)
    return unseen_movie_indices.flatten()

def top_user_similarity(data, user_id, top_n=5):
    
    user_id -= 1
    unseen_movies = find_unseen_movies(user_id)
    
    # initialize closest users list
    closest_users = [(None, -np.inf)] * top_n
    
    # movie-vector
    x =  data[user_id, :]
    
    for user, y in enumerate(data):
        
        if np.linalg.norm(y) != 0: similarity = cosine_similarity(x, y)
        else: similarity = 0

        try:
            best_index = next(ind for ind, val in enumerate(closest_users) if val[1] < similarity)
            
            # if there is a value in the list which is less than this similarity, then add
            # the user ID and the similarity to the list and remove the final item in the list
            closest_users.insert(best_index, (user, similarity))
            closest_users.pop(-1)
            
        except StopIteration:
            # if this similarity is not in the top_n closest_movies just move on
            pass
    
    movie_recommendations = [(None, -np.inf)] * top_n
    closest_user_id = closest_users[0][0]
    
    for movie_ind in unseen_movies:
        rating = data[closest_user_id, movie_ind]

        if rating == 0:
            pass
        else:
            try:
                best_index = next(ind for ind, val in enumerate(movie_recommendations) if val[1] < rating)

                # if there is a value in the list which is less than this similarity, then add
                # the movie ID and the similarity to the list and remove the final item in the list
                movie_recommendations.insert(best_index, (movie_ind, rating))
                movie_recommendations.pop(-1)
            
            except StopIteration:
                # if this similarity is not in the top_n closest_movies just move on
                pass
    
    return movie_recommendations


# In[31]:


movie_recommendations = top_user_similarity(R1000, 5954, top_n=5)
for rank, (movie_ind, rating) in enumerate(movie_recommendations):
    print("%i) %s --> Rating: %1.2f" % (rank+1, movie_titles[movie_ind+1], rating))


# In[ ]:




