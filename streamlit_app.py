import streamlit as st
from model import get_recipe_recommendations
import pandas as pd


st.title('Recipe Recommendation System')
st.text('Provide ingredients name by commas separated')
ingred = st.text_input(
    'Enter the leftover ingredients separated by commas: ')
if st.button('Recommend'):
    recommendations = get_recipe_recommendations(ingred)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(recommendations)

    # Display the DataFrame in Streamlit

    # Pre-process the DataFrame
    df['cosine_sim'] = df['cosine_sim'] * 100
    
    # st.dataframe(df)
    st.data_editor(df, column_config={
        'rank': st.column_config.NumberColumn(
            "Rank", format="%.0f"),
        'recipe_name': st.column_config.TextColumn(
            "Recipe Name"),
        'cosine_sim': st.column_config.NumberColumn(
            "Similarity", format="%.2f%%"),  # Modify the format to display as percentage
        'img_src': st.column_config.ImageColumn(
            "Preview Image"),
        'url': st.column_config.LinkColumn(
            "Link", display_text="Open Recipe's link")
    })

# leftover_ingredients = st.text_input('Enter the leftover ingredients separated by commas: ')
# recommendations = get_recipe_recommendations(leftover_ingredients)
# st.write(recommendations)
