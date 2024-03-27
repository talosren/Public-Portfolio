import pandas as pd

# Read in the data
schools = pd.read_csv("schools.csv")

# Preview the data
schools.head()

# Create a pandas DataFrame called best_math_schools containing the "school_name" and "average_math" score for all schools where the results are at least 80% of the maximum possible score, sorted by "average_math" in descending order.
# Schools subset
# 80% when the maximum score is 800 -> 640
best_math_schools = schools[schools["average_math"] >= 640][["school_name", "average_math"]].sort_values(by = "average_math", ascending = False)                                
best_math_schools.head(5)

# Identify the top 10 performing schools based on scores across the three SAT sections, storing as a pandas DataFrame called top_10_schools containing the school name and a column named "total_SAT", with results sorted by total_SAT in descending order.
schools["total_SAT"] = schools["average_math"] + schools["average_reading"] + schools["average_writing"]
top_10_schools = schools.groupby("school_name")["total_SAT"].mean().reset_index().sort_values(by = "total_SAT", ascending = False).head(10)
top_10_schools.head(10)

# Locate the NYC borough with the largest standard deviation for "total_SAT", storing as a DataFrame called largest_std_dev with "borough" as the index and three columns: "num_schools" for the number of schools in the borough, "average_SAT" for the mean of "total_SAT", and "std_SAT" for the standard deviation of "total_SAT". Round all numeric values to two decimal places.

# Making subset
boroughs = schools.groupby("borough")["total_SAT"].agg(["count", "mean", "std"]).round(2)

# Making the largest known
largest_std_dev = boroughs[boroughs["std"] == boroughs["std"].max()]

# Renaming
largest_std_dev = largest_std_dev.rename(columns = {"count":"num_schools", "mean":"average_SAT", "std":"std_SAT"})
largest_std_dev.head()
