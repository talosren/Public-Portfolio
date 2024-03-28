import pandas as pd
import numpy as np

# Categories year,category,prize,motivation,prize_share,laureate_id,laureate_type,full_name,birth_date,birth_city,birth_country,sex,organization_name,organization_city,organization_country,death_date,death_city,death_country

# Read
df = pd.read_csv('Python Projects\data\nobel.csv')

# Q1: What is the most commonly awarded gender and birth country?
df.head(10) # -> Looking into categories

top_gender = df['sex'].value_counts().index[0]
top_country = df['birth_country'].value_counts().index[0]

print(f"Most Commonly Awarded Gender: {top_gender}")
print(f"Most Commonly Awarded Birth Country: {top_country}")

# ----------------------Answer---------------------------------
# Most Commonly Awarded Gender: Male
# Most Commonly Awarded Birth Country: United States of America
# -------------------------------------------------------------

# Q2: Which decade had the highest ratio of US-born Nobel Prize winners to total winners in all categories?

# Proportions

df['usa_winner'] = df['birth_country'] == 'United States of America'
df['decade'] = (np.floor(df['year'] / 10) * 10).astype(int)

temp = df.groupby('decade', as_index = False)['usa_winner'].mean()

max_decade_usa = temp[temp['usa_winner'] == temp['usa_winner'].max()]['decade'].values[0]

print(f"Decade of Highest Ratio: {max_decade_usa}")

# ----------Answer-------------
# Decade of Highest Ratio: 2000
# -----------------------------

# Q3: Which decade and Nobel Prize category combination had the highest proportion of female laureates?

# Store in dictionary as {decade : category}

df['female_winner'] = df['sex'] == 'Female'

temp = df.groupby(['decade', 'category'], as_index = False)['female_winner'].mean()

key_val = temp[temp['female_winner'] == temp['female_winner'].max()][['decade', 'category']]

# max_female_dict = {max_female_decade_category['decade'].values[0]: max_female_decade_category['category'].values[0]}

max_female_dict = {key_val['decade'].values[0] : key_val['category'].values[0]}

print(f"Decade : Nobel Prize: {max_female_dict}")

# ----------------Answer--------------------
# Decade : Nobel Prize: {2020: 'Literature'}
# ------------------------------------------

# Q4: Who was the first woman to receive a Nobel Prize, and in what category?
first_woman = df[df['female_winner']]
year = first_woman[first_woman['year'] == first_woman['year'].min()]

first_woman_name = year['full_name'].values[0]
first_woman_category = year['category'].values[0]

print(f"First Woman Name: {first_woman_name}")
print(f"First Woman Category: {first_woman_category}")

# -----------------Answer----------------------
# First Woman Name: Marie Curie, née Sklodowska
# First Woman Category: Physics
# ---------------------------------------------

# Q5: Which individuals or organizations have won more than one Nobel Prize throughout the years?

counts = df['full_name'].value_counts()
repeats = counts[counts >= 2].index
repeat_list = list(repeats)
    
print(f"Repeat Winners: {repeat_list}")


# -----------------------------------------------Answer---------------------------------------------------
# Repeat Winners: ['Comité international de la Croix Rouge (International Committee of the Red Cross)', 
#                 'Linus Carl Pauling', 'John Bardeen', 'Frederick Sanger', 'Marie Curie, née Sklodowska', 
#                 'Office of the United Nations High Commissioner for Refugees (UNHCR)']
# --------------------------------------------------------------------------------------------------------


# Q1: Most Commonly Awarded Gender: Male
# Q1: Most Commonly Awarded Birth Country: United States of America
# Q2: Decade of Highest Ratio: 2000
# Q3: Decade : Nobel Prize: {2020: 'Literature'}
# Q4: First Woman Name: Marie Curie, née Sklodowska
# Q4: First Woman Category: Physics
# Q5: Repeat Winners: ['Comité international de la Croix Rouge (International Committee of the Red Cross)', 'Linus Carl Pauling', 'John Bardeen', 'Frederick Sanger', 'Marie Curie, née Sklodowska', 'Office of the United Nations High Commissioner for Refugees (UNHCR)']