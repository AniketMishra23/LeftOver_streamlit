import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re

# Load the recipe dataset
recipes_df = pd.read_csv('recipes.csv')

# Clean the dataset by removing unnecessary columns
recipes_df.drop(columns=["prep_time", "cook_time", "total_time", "yield", "rating",
                         "cuisine_path", "nutrition"], inplace=True)

# Create a bag of words for the ingredients column using only food names
cv = CountVectorizer(stop_words='english', token_pattern=r'\b[A-Za-z]+\b')
ingredients_matrix = cv.fit_transform(recipes_df['ingredients'])

# Calculate the cosine similarity between the ingredients matrix
cosine_sim = cosine_similarity(ingredients_matrix)

def get_recipe_recommendations(leftover_ingredients):
    if leftover_ingredients is None:
        return []

    # Transform the leftover ingredients to contain only food names
    leftover_ingredients = ', '.join(re.findall(r'\b[A-Za-z]+\b', leftover_ingredients))

    # Transform the recipes ingredients to contain only food names
    recipes_df['ingredients'] = recipes_df['ingredients'].apply(
        lambda x: ', '.join(re.findall(r'\b[A-Za-z]+\b', x))
    )

    # Calculate cosine similarity
    ingredients_matrix = cv.transform(recipes_df['ingredients'])
    cosine_similarities = cosine_similarity(ingredients_matrix, cv.transform([leftover_ingredients]))

    # Add cosine_sim column to recipes_df
    recipes_df['cosine_sim'] = cosine_similarities.flatten()

    # Get recipe recommendations
    sorted_df = recipes_df.sort_values('cosine_sim', ascending=False).reset_index()
    sorted_df['rank'] = sorted_df.index + 1  # Add rank column starting from 1
    recommendations = sorted_df[['rank', 'recipe_name', 'cosine_sim', 'url', 'img_src']].head(10)

    return recommendations.to_dict(orient='records')

# # Take user input for leftover ingredients
# leftover_ingredients = input("Enter the leftover ingredients separated by commas: ")
# recommendations = get_recipe_recommendations(leftover_ingredients)
# print(recommendations)

# # Save the model
# joblib.dump(cosine_sim, 'recipe_rec.joblib')
