import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import pickle
import os

st.set_page_config(layout="wide", page_icon="🎞",
                   page_title="Movie Recommendation")

stss = st.session_state

df = pd.read_csv("cleaned_data/movies_metadata_small_cleaned.csv")

if "columns_loaded" not in stss:
    stss.columns_loaded = {}
    with open("cleaned_data/genres_cols.pickle", "rb") as f:
        stss.columns_loaded["genres_cols"] = pickle.load(f)
    with open("cleaned_data/spoken_languages_cols.pickle", "rb") as f:
        stss.columns_loaded["spoken_languages_cols"] = pickle.load(f)
    with open("cleaned_data/cast_cols_top1000.pickle", "rb") as f:
        stss.columns_loaded["cast_cols"] = pickle.load(f)
    with open("cleaned_data/production_companies_cols_top500.pickle", "rb") as f:
        stss.columns_loaded["production_companies_cols"] = pickle.load(f)


def filter_df():
    common_idxs = list(stss.all_filters.values())[0].index
    for filtered_df in list(stss.all_filters.values())[1:]:
        common_idxs = common_idxs.intersection(filtered_df.index)
    stss.filtered_df = df.loc[common_idxs]


def movie_counter(matrix, unique_cols, unfiltered_input=False, idxs_to_filter=None):
    assert (unfiltered_input and idxs_to_filter is not None) or (not unfiltered_input and idxs_to_filter is None), \
        "If the input matrix is unfiltered, provide the indeces required to filter it."
    if unfiltered_input:
        matrix = matrix[idxs_to_filter, :]
    movie_counter = {}
    for i, attribute in enumerate(unique_cols):
        movie_counter[attribute] = matrix[:, i].getnnz()
    return movie_counter


def recount_movies():
    stss.movie_label_counter['genres'] = movie_counter(stss.genres_matrix,
                                                       stss.columns_loaded["genres_cols"],
                                                       unfiltered_input=True,
                                                       idxs_to_filter=list(stss.filtered_df.index))
    stss.movie_label_counter['spoken_languages'] = movie_counter(stss.spoken_languages_matrix,
                                                                 stss.columns_loaded["spoken_languages_cols"],
                                                                 unfiltered_input=True,
                                                                 idxs_to_filter=list(stss.filtered_df.index))
    stss.movie_label_counter['cast'] = movie_counter(stss.cast_matrix,
                                                     stss.columns_loaded["cast_cols"],
                                                     unfiltered_input=True,
                                                     idxs_to_filter=list(stss.filtered_df.index))
    stss.movie_label_counter['production_companies'] = movie_counter(stss.production_companies_matrix,
                                                                     stss.columns_loaded["production_companies_cols"],
                                                                     unfiltered_input=True,
                                                                     idxs_to_filter=list(stss.filtered_df.index))

# def count_genres(genres_matrix, dense_input_matrix=True):
#    if dense_input_matrix:
#        genres_matrix = stss.genres_matrix[list(genres_matrix.index),:]
#    genres_counter = {}
#    for i, genre in enumerate(genres_cols):
#        genres_counter[genre] = genres_matrix[:,i].getnnz()
#    return genres_counter


def filter_by_title():
    if stss.title_selection is not None:
        stss.filtered_df = df[df['original_title']
                              == stss.title_selection]
    else:
        stss.filtered_df = df


def update_sparse_matrix(session_state, filtered_matrix, unique_cols):
    if len(session_state) > 0:
        for attribute in session_state:
            attribute_col_idx = unique_cols.index(attribute)
            attribute_col = filtered_matrix[:, attribute_col_idx]
            filtered_matrix = filtered_matrix.multiply(attribute_col)
    return filtered_matrix


def filter_by_genre():
    stss.filtered_genres_matrix = stss.genres_matrix
    stss.filtered_genres_matrix = update_sparse_matrix(stss.genre_selection,
                                                       stss.filtered_genres_matrix,
                                                       stss.columns_loaded["genres_cols"])
    movie_idxs = np.flatnonzero(np.diff(stss.filtered_genres_matrix.indptr))
    movie_idxs = np.unique(stss.filtered_genres_matrix.nonzero()[0])
    stss.all_filters["filter_by_genre"] = df.iloc[movie_idxs]
    filter_df()
    recount_movies()


def filter_by_year():
    stss.all_filters["filter_by_year"] = df[(df['release_date'] >= stss.year_selection[0]) &
                                            (df['release_date'] <= stss.year_selection[1])]
    filter_df()
    recount_movies()


def filter_by_spoken_language():
    stss.filtered_spoken_languages_matrix = stss.spoken_languages_matrix
    stss.filtered_spoken_languages_matrix = update_sparse_matrix(stss.spoken_language_selection,
                                                                 stss.filtered_spoken_languages_matrix,
                                                                 stss.columns_loaded["spoken_languages_cols"])
    movie_idxs = np.flatnonzero(
        np.diff(stss.filtered_spoken_languages_matrix.indptr))
    stss.all_filters["filter_by_spoken_language"] = df.iloc[movie_idxs]
    filter_df()
    recount_movies()


def filter_by_cast():
    stss.filtered_cast_matrix = stss.cast_matrix
    stss.filtered_cast_matrix = update_sparse_matrix(stss.cast_selection,
                                                     stss.filtered_cast_matrix,
                                                     stss.columns_loaded["cast_cols"])
    movie_idxs = np.flatnonzero(np.diff(stss.filtered_cast_matrix.indptr))
    stss.all_filters["filter_by_cast"] = df.iloc[movie_idxs]
    filter_df()
    recount_movies()


def filter_by_production_company():
    stss.filtered_production_companies_matrix = stss.production_companies_matrix
    stss.filtered_production_companies_matrix = update_sparse_matrix(stss.production_company_selection,
                                                                     stss.filtered_production_companies_matrix,
                                                                     stss.columns_loaded["production_companies_cols"])
    movie_idxs = np.flatnonzero(
        np.diff(stss.filtered_production_companies_matrix.indptr))
    stss.all_filters["filter_by_production_company"] = df.iloc[movie_idxs]
    filter_df()
    recount_movies()


if "genres_matrix" not in stss:
    stss.genres_matrix = load_npz("cleaned_data/genres_matrix.npz")
if "spoken_languages_matrix" not in stss:
    stss.spoken_languages_matrix = load_npz(
        "cleaned_data/spoken_languages_matrix.npz")
if "cast_matrix" not in stss:
    stss.cast_matrix = load_npz("cleaned_data/cast_matrix_top1000.npz")
if "production_companies_matrix" not in stss:
    stss.production_companies_matrix = load_npz(
        "cleaned_data/production_companies_matrix_top500.npz")
if "movie_label_counter" not in stss:
    stss.movie_label_counter = {}
    stss.movie_label_counter['genres'] = movie_counter(stss.genres_matrix,
                                                       stss.columns_loaded["genres_cols"])
    stss.movie_label_counter['spoken_languages'] = movie_counter(stss.spoken_languages_matrix,
                                                                 stss.columns_loaded["spoken_languages_cols"])
    stss.movie_label_counter['cast'] = movie_counter(stss.cast_matrix,
                                                     stss.columns_loaded["cast_cols"])
    stss.movie_label_counter['production_companies'] = movie_counter(stss.production_companies_matrix,
                                                                     stss.columns_loaded["production_companies_cols"])
if "all_filters" not in stss:
    stss.all_filters = {}
if "filtered_df" not in stss:
    stss.filtered_df = df


with st.sidebar:

    st.markdown("""
            <style>
            h1 {
                padding-top: 10px;
            }
            [data-testid="stSidebar"][aria-expanded="true"] {
                min-width: 25rem;
                max-width: 25rem;
            }
            header[data-testid="stHeader"] {
                display: none;
            }
            div[data-testid="stSidebarHeader"] {
                padding-bottom: 0;
            }
            div[data-testid="stSidebarUserContent"] {
                align-items: top;
                padding-bottom: 48px;
                padding-top: 8px;
            }
            section[data-testid="stSidebar] {
                width: 450px;
                height: 100%;
            }
            </style>
            """,
                unsafe_allow_html=True)

    st.write("# Search By Title")

    st.selectbox(label="Title Search",
                 label_visibility="collapsed",
                 index=None,
                 options=sorted(df['original_title']),
                 on_change=filter_by_title,
                 key="title_selection",
                 placeholder="Write the title of the movie")

    st.markdown("---")

    st.write("# Search By Filter")

    st.write("### Genres")
    st.multiselect(label="Genres",
                   label_visibility='collapsed',
                   options=dict(sorted(stss.movie_label_counter['genres'].items(
                   ), key=lambda counter: counter[1], reverse=True)),
                   format_func=lambda genre: f"{
                       genre} ({stss.movie_label_counter['genres'][genre]})",
                   default=stss.genre_selection if "genre_selection" in stss else None,
                   on_change=filter_by_genre,
                   key='genre_selection')

    st.write("### Release Year")
    if "year_selection" not in stss:
        stss.year_selection = (
            min(df['release_date']), max(df['release_date']))
    st.slider(label="Release Year",
                    label_visibility='collapsed',
                    min_value=min(df['release_date']),
                    max_value=max(df['release_date']),
                    on_change=filter_by_year,
                    key='year_selection')

    st.write("### Spoken Languages")
    st.multiselect(label="Spoken Languages",
                   label_visibility="collapsed",
                   options=dict(sorted(stss.movie_label_counter['spoken_languages'].items(
                   ), key=lambda counter: counter[1], reverse=True)),
                   format_func=lambda spoken_language: f"{
                       spoken_language} ({stss.movie_label_counter['spoken_languages'][spoken_language]})",
                   default=stss.spoken_language_selection if "spoken_language_selection" in stss else None,
                   on_change=filter_by_spoken_language,
                   key="spoken_language_selection")

    st.write("### Cast Members")
    st.multiselect(label="Cast Members",
                   label_visibility="collapsed",
                   options=dict(sorted(stss.movie_label_counter['cast'].items(
                   ), key=lambda counter: counter[1], reverse=True)),
                   format_func=lambda cast: f"{
                       cast} ({stss.movie_label_counter['cast'][cast]})",
                   default=stss.cast_selection if "cast_selection" in stss else None,
                   on_change=filter_by_cast,
                   key="cast_selection")

    st.write("### Production Companies")
    st.multiselect(label="Production Companies",
                   label_visibility="collapsed",
                   options=dict(sorted(stss.movie_label_counter['production_companies'].items(),
                                       key=lambda counter: counter[1],
                                       reverse=True)),
                   format_func=lambda production_companies: f"{production_companies} ({
                       stss.movie_label_counter['production_companies'][production_companies]})",
                   default=stss.production_company_selection if "production_company_selection" in stss else None,
                   on_change=filter_by_production_company,
                   key="production_company_selection")

    def reset_filters():
        stss.filtered_df = df
        for state in stss:
            stss.pop(state)

    st.write("")
    st.button(label="Reset Filters",
              type="secondary",
              on_click=reset_filters,
              key="reset_filters")


stss.filtered_df  # TESTING ---------------------


st.write("## Top 20 Movies")
# st.selectbox(label="Sort by:")
first_movie_row = st.columns(5)
second_movie_row = st.columns(5)
third_movie_row = st.columns(5)
fourth_movie_row = st.columns(5)
posters = [f for f in os.listdir("assets/posters/")]
for i, col in enumerate(first_movie_row):
    col.image("assets/posters/" + posters[i])
for i, col in enumerate(second_movie_row):
    col.image("assets/posters/" + posters[i+5])
for i, col in enumerate(third_movie_row):
    col.image("assets/posters/" + posters[i+10])
for i, col in enumerate(fourth_movie_row):
    col.image("assets/posters/" + posters[i+15])


# TODO: NUMBER OF MOVIES NOT CORRECT WHEN FILTERS ARE REMOVED (NOT SURE WHEN RESET) - HAPPENS AT movies_idxs
