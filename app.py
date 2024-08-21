import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import ast
import gc

st.markdown("""
            <style>
            div[aria-label="Spoken Language Checkbox"]  {
                max-height: 200px;
                overflow: auto;
            }

            section[data-testid=stSidebar] {
                width: 400px;
            }
            </style>
            """,
            unsafe_allow_html=True)

with st.sidebar:
    with st.form(key="movie_filters", clear_on_submit=True, border=False):
        st.write("### Title Search")
        st.selectbox(label="Movie Title Search", 
                            label_visibility="collapsed", 
                            index=None,
                            options=["English", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German"],
                            key="movie_title")

        st.write("### Genres")
        st.multiselect(label="Genre", 
                            label_visibility='collapsed', 
                            options=[f'Drama ()', f'Horror ()', f'Action ()'],
                            default=None,
                            key='genre_selection')

        st.write("### Year")
        st.slider(label="Year", 
                        label_visibility='collapsed', 
                        min_value=2000, 
                        max_value=2020, 
                        value=[2000, 2020], 
                        key='year_selection')

        st.write("### Spoken Language")
        st.radio(label="Spoken Language Checkbox",
                 label_visibility="collapsed",
                 index=0, 
                 options=["English", "Portuguese", "French", "German", "Portuguese", "French", "German", "Portuguese", "French", "German"],
                 key="spoken_language")

        st.form_submit_button("Clear Filters")

st.write("## HELLO!")