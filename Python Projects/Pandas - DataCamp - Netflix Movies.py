import pandas as pd
import matplotlib.pyplot as plt

# Creating the dataframe
netflix_df = pd.read_csv('netflix_data.csv')
# print(netflix_df) # 7787 Rows x 11 Columns

# Q1: Filter the data to remove TV shows and store as netflix_subset.
# Creating a subset that only shows data that is a TV Show
netflix_subset = netflix_df[netflix_df['type'] != 'TV Show']
# print(netflix_subset) # 5377 Rows x 11 Columns

# Investigate the Netflix movie data, 
# keeping only the columns "title", "country", "genre", "release_year", "duration", and saving this into a new DataFrame called netflix_movies.
netflix_movies = netflix_subset[['title', 'country', 'genre', "release_year", "duration"]]
# print(netflix_movies) # 7787 Rows x 5 Columns

# Filter netflix_movies to find the movies that are strictly shorter than 60 minutes, 
# saving the resulting DataFrame as short_movies; 
# inspect the result to find possible contributing factors.
short_movies = netflix_movies[netflix_movies.duration < 60]
# 2830 Rows x 5 Columns

# Using a for loop and if/elif statements, 
# iterate through the rows of netflix_movies and assign colors of your choice to four genre groups 
# ("Children", "Documentaries", "Stand-Up", and "Other" for everything else). 
# Save the results in a colors list. 
colors = []

for index, row in netflix_movies.iterrows():
    if row['genre'] == 'Children':
        colors.append("Blue")
    elif row['genre'] == "Documenteries":
        colors.append("Green")
    elif row['genre'] == "Stand-Up":
        colors.append("Orange")
    else:
        colors.append("Brown")

# Initialize a matplotlib figure object called fig and create a scatter plot for movie duration 
# by release year using the colors list to color the points and using the labels 
# "Release year" for the x-axis, "Duration (min)" for the y-axis,
#  and the title "Movie Duration by Year of Release".
fig = plt.figure(figsize = (12, 8))

# x - Release Year
x = netflix_movies['release_year']

# y - Duration
y = netflix_movies['duration']

plt.scatter(x ,y, c = colors)

plt.xlabel("Release year")
plt.ylabel("Duration (min)")
plt.title("Movie Duration by Year of Release")
plt.show()