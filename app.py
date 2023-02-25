#!/usr/bin/env python
# coding: utf-8

# In[14]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# Step 1: Load the dataset
data = pd.read_csv('product_images.csv')
data = data.iloc[1:].values.astype(float)

# Step 2: Apply PCA
pca = PCA(n_components=60)
data_pca = pca.fit_transform(data)

# Step 3: Apply K-means clustering
kmeans = KMeans(n_clusters=5)
kmeans.fit(data_pca)
labels = kmeans.labels_

# Step 4: Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(data_pca)

# Step 5: Define function to find top similar images
def find_similar_images(image_index, n=9):
    # Get similarity scores for all images
    scores = cosine_sim_matrix[image_index]
    # Sort scores in descending order and get top n+1 indexes
    top_n_indexes = np.argsort(-scores)[:n+1]
    # Remove image_index from top_n_indexes
    top_n_indexes = top_n_indexes[top_n_indexes != image_index]
    # Return top n images
    return data[top_n_indexes]



# Step 6: Create Streamlit app
st.set_page_config(page_title="ManSci Shop", page_icon="🛍️", initial_sidebar_state="expanded")
st.title('🛍️ManSci Shop🛍️')
st.markdown('''##### <span style="color:gray">Give Daily Recommendation</span>
            ''', unsafe_allow_html=True)

    # Sidebar
col1, col2, col3 = st.sidebar.columns([1,8,1])
with col1:
    st.write("")
with col2:
    st.image('sidebar.jpeg',  use_column_width=True)
with col3:
    st.write("")
st.sidebar.markdown(" ## About ManSci Shop")
st.sidebar.markdown("This is UCL ManSci Shop, and this App can recommend similar products to the items the customers are viewing."  )              
st.sidebar.info("made by CH in ManSci Data SW")



if st.button("Today's Recommendation"):
    st.write("click again to get another recommendation")
    # Select random image
    random_index = np.random.randint(len(data))
    image = data[random_index].reshape(28, 28) / 255.0
    # Display random image
    st.image(image, caption="Today's Recommendation", width=450)


    # Display text with larger font size using CSS styling
    st.markdown("<p style='font-size: 24px;'>Here are some similar products:</p>", unsafe_allow_html=True)

    
    # Find top 9 similar images excluding the random image
    similar_images = find_similar_images(random_index)
  

    # Define three equal-width columns
    col1, col2, col3 = st.columns(3)
    

    # Display the first 3 similar images in the first column
    with col1:
        for i in range(3):
            image = similar_images[i].reshape(28, 28) / 255.0
            st.image(image, use_column_width=True)

    # Display the next 3 similar images in the second column
    with col2:
        for i in range(3, 6):
            image = similar_images[i].reshape(28, 28) / 255.0
            st.image(image, use_column_width=True)

    # Display the last 3 similar images in the third column
    with col3:
        for i in range(6, 9):
            image = similar_images[i].reshape(28, 28) / 255.0
            st.image(image, use_column_width=True)



# In[ ]:




