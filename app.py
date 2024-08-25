import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import pickle
import os

st.set_page_config(layout="wide", page_icon="🎞",
                   page_title="Movie Recommendation")


@st.cache_data
def get_data():
    df = pd.read_csv("cleaned_data/movies_metadata_small_cleaned.csv")
    with open("cleaned_data/genres_cols.pickle", "rb") as f:
        genres_cols = pickle.load(f)
    with open("cleaned_data/spoken_languages_cols.pickle", "rb") as f:
        spoken_languages_cols = pickle.load(f)
    with open("cleaned_data/cast_cols_top1000.pickle", "rb") as f:
        cast_cols = pickle.load(f)
    with open("cleaned_data/production_companies_cols_top500.pickle", "rb") as f:
        production_companies_cols = pickle.load(f)
    return df, genres_cols, spoken_languages_cols, cast_cols, production_companies_cols


df, genres_cols, spoken_languages_cols, cast_cols, production_companies_cols = get_data()


def filter_df():
    st.session_state.disable_title_search = True
    common_idxs = df.index
    for filtered_df in list(st.session_state.filters.values()):
        common_idxs = common_idxs.intersection(filtered_df.index)
    st.session_state.displayed_df = df.iloc[common_idxs].copy()


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
    states = [st.session_state.genre_selection,
              st.session_state.year_selection != (
                  min(df['release_date']), max(df['release_date'])),
              st.session_state.spoken_language_selection,
              st.session_state.cast_selection,
              st.session_state.production_company_selection]
    if not any(states):
        st.session_state.disable_title_search = False
    st.session_state.movie_label_counter['genres'] = movie_counter(st.session_state.genres_matrix,
                                                                   genres_cols,
                                                                   unfiltered_input=True,
                                                                   idxs_to_filter=list(st.session_state.displayed_df.index))
    st.session_state.movie_label_counter['spoken_languages'] = movie_counter(st.session_state.spoken_languages_matrix,
                                                                             spoken_languages_cols,
                                                                             unfiltered_input=True,
                                                                             idxs_to_filter=list(st.session_state.displayed_df.index))
    st.session_state.movie_label_counter['cast'] = movie_counter(st.session_state.cast_matrix,
                                                                 cast_cols,
                                                                 unfiltered_input=True,
                                                                 idxs_to_filter=list(st.session_state.displayed_df.index))
    st.session_state.movie_label_counter['production_companies'] = movie_counter(st.session_state.production_companies_matrix,
                                                                                 production_companies_cols,
                                                                                 unfiltered_input=True,
                                                                                 idxs_to_filter=list(st.session_state.displayed_df.index))


def filter_by_title():
    if st.session_state.title_selection is not None:
        st.session_state.disable_filters = True
        st.session_state.displayed_df = df[df['original_title']
                                           == st.session_state.title_selection].copy()
    else:
        st.session_state.displayed_df = df.copy()
        st.session_state.disable_filters = False


def update_sparse_matrix(selection, filtered_matrix, unique_cols):
    for attribute in selection:
        attribute_col_idx = unique_cols.index(attribute)
        attribute_col = filtered_matrix[:, attribute_col_idx]
        filtered_matrix = filtered_matrix.multiply(attribute_col)
    return filtered_matrix


def filter_by_genre():
    if len(st.session_state.genre_selection) > 0:
        st.session_state.filtered_genres_matrix = st.session_state.genres_matrix
        st.session_state.filtered_genres_matrix = update_sparse_matrix(st.session_state.genre_selection,
                                                                       st.session_state.filtered_genres_matrix,
                                                                       genres_cols)
        movie_idxs = np.flatnonzero(
            np.diff(st.session_state.filtered_genres_matrix.indptr))
        st.session_state.filters["filter_by_genre"] = df.iloc[movie_idxs].copy(
        )
    else:
        st.session_state.filters.pop("filter_by_genre")
    filter_df()
    recount_movies()


def filter_by_year():
    st.session_state.filters["filter_by_year"] = df[(df['release_date'] >= st.session_state.year_selection[0]) &
                                                    (df['release_date'] <= st.session_state.year_selection[1])].copy()
    filter_df()
    recount_movies()


def filter_by_spoken_language():
    if len(st.session_state.spoken_language_selection) > 0:
        st.session_state.filtered_spoken_languages_matrix = st.session_state.spoken_languages_matrix
        st.session_state.filtered_spoken_languages_matrix = update_sparse_matrix(st.session_state.spoken_language_selection,
                                                                                 st.session_state.filtered_spoken_languages_matrix,
                                                                                 spoken_languages_cols)
        movie_idxs = np.flatnonzero(
            np.diff(st.session_state.filtered_spoken_languages_matrix.indptr))
        st.session_state.filters["filter_by_spoken_language"] = df.iloc[movie_idxs].copy(
        )
    else:
        st.session_state.filters.pop("filter_by_spoken_language")
    filter_df()
    recount_movies()


def filter_by_cast():
    if len(st.session_state.cast_selection) > 0:
        st.session_state.filtered_cast_matrix = st.session_state.cast_matrix
        st.session_state.filtered_cast_matrix = update_sparse_matrix(st.session_state.cast_selection,
                                                                     st.session_state.filtered_cast_matrix,
                                                                     cast_cols)
        movie_idxs = np.flatnonzero(
            np.diff(st.session_state.filtered_cast_matrix.indptr))
        st.session_state.filters["filter_by_cast"] = df.iloc[movie_idxs].copy()
    else:
        st.session_state.filters.pop("filter_by_cast")
    filter_df()
    recount_movies()


def filter_by_production_company():
    if len(st.session_state.production_company_selection) > 0:
        st.session_state.filtered_production_companies_matrix = st.session_state.production_companies_matrix
        st.session_state.filtered_production_companies_matrix = update_sparse_matrix(st.session_state.production_company_selection,
                                                                                     st.session_state.filtered_production_companies_matrix,
                                                                                     production_companies_cols)
        movie_idxs = np.flatnonzero(
            np.diff(st.session_state.filtered_production_companies_matrix.indptr))
        st.session_state.filters["filter_by_production_company"] = df.iloc[movie_idxs].copy(
        )
    else:
        st.session_state.filters.pop("filter_by_production_company")
    filter_df()
    recount_movies()


if "genres_matrix" not in st.session_state:
    st.session_state.genres_matrix = load_npz("cleaned_data/genres_matrix.npz")
if "spoken_languages_matrix" not in st.session_state:
    st.session_state.spoken_languages_matrix = load_npz(
        "cleaned_data/spoken_languages_matrix.npz")
if "cast_matrix" not in st.session_state:
    st.session_state.cast_matrix = load_npz(
        "cleaned_data/cast_matrix_top1000.npz")
if "production_companies_matrix" not in st.session_state:
    st.session_state.production_companies_matrix = load_npz(
        "cleaned_data/production_companies_matrix_top500.npz")
if "movie_label_counter" not in st.session_state:
    st.session_state.movie_label_counter = {}
    st.session_state.movie_label_counter['genres'] = movie_counter(st.session_state.genres_matrix,
                                                                   genres_cols)
    st.session_state.movie_label_counter['spoken_languages'] = movie_counter(st.session_state.spoken_languages_matrix,
                                                                             spoken_languages_cols)
    st.session_state.movie_label_counter['cast'] = movie_counter(st.session_state.cast_matrix,
                                                                 cast_cols)
    st.session_state.movie_label_counter['production_companies'] = movie_counter(st.session_state.production_companies_matrix,
                                                                                 production_companies_cols)
if "disable_filters" not in st.session_state:
    st.session_state.disable_filters = False
if "disable_title_search" not in st.session_state:
    st.session_state.disable_title_search = False
if "filters" not in st.session_state:
    st.session_state.filters = {}
if "displayed_df" not in st.session_state:
    st.session_state.displayed_df = df.copy()


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
                 placeholder="Type the title of the movie",
                 disabled=st.session_state.disable_title_search)

    st.markdown("---")

    st.write("# Search By Filter")

    st.write("### Genres")
    st.multiselect(label="Genres",
                   label_visibility='collapsed',
                   options=dict(sorted(st.session_state.movie_label_counter['genres'].items(
                   ), key=lambda counter: counter[1], reverse=True)),
                   format_func=lambda genre: f"{
                       genre} ({st.session_state.movie_label_counter['genres'][genre]})",
                   default=st.session_state.genre_selection if "genre_selection" in st.session_state else None,
                   on_change=filter_by_genre,
                   key='genre_selection',
                   disabled=st.session_state.disable_filters)

    st.write("### Release Year")
    if "year_selection" not in st.session_state:
        st.session_state.year_selection = (
            min(df['release_date']), max(df['release_date']))
    st.slider(label="Release Year",
                    label_visibility='collapsed',
                    min_value=min(df['release_date']),
                    max_value=max(df['release_date']),
                    on_change=filter_by_year,
                    key='year_selection',
                    disabled=st.session_state.disable_filters)

    st.write("### Spoken Languages")
    st.multiselect(label="Spoken Languages",
                   label_visibility="collapsed",
                   options=dict(sorted(st.session_state.movie_label_counter['spoken_languages'].items(
                   ), key=lambda counter: counter[1], reverse=True)),
                   format_func=lambda spoken_language: f"{
                       spoken_language} ({st.session_state.movie_label_counter['spoken_languages'][spoken_language]})",
                   default=st.session_state.spoken_language_selection if "spoken_language_selection" in st.session_state else None,
                   on_change=filter_by_spoken_language,
                   key="spoken_language_selection",
                   disabled=st.session_state.disable_filters)

    st.write("### Cast Members")
    st.multiselect(label="Cast Members",
                   label_visibility="collapsed",
                   options=dict(sorted(st.session_state.movie_label_counter['cast'].items(
                   ), key=lambda counter: counter[1], reverse=True)),
                   format_func=lambda cast: f"{
                       cast} ({st.session_state.movie_label_counter['cast'][cast]})",
                   default=st.session_state.cast_selection if "cast_selection" in st.session_state else None,
                   on_change=filter_by_cast,
                   key="cast_selection",
                   disabled=st.session_state.disable_filters)

    st.write("### Production Companies")
    st.multiselect(label="Production Companies",
                   label_visibility="collapsed",
                   options=dict(sorted(st.session_state.movie_label_counter['production_companies'].items(),
                                       key=lambda counter: counter[1],
                                       reverse=True)),
                   format_func=lambda production_companies: f"{production_companies} ({
                       st.session_state.movie_label_counter['production_companies'][production_companies]})",
                   default=st.session_state.production_company_selection if "production_company_selection" in st.session_state else None,
                   on_change=filter_by_production_company,
                   key="production_company_selection",
                   disabled=st.session_state.disable_filters)

    def reset_filters():
        st.session_state.displayed_df = df
        for state in st.session_state:
            st.session_state.pop(state)

    st.write("")
    st.button(label="Reset Filters",
              type="secondary",
              on_click=reset_filters,
              key="reset_filters",
              disabled=st.session_state.disable_filters)


# TESTING --------------------- #

def sort_df():
    sorting_methods = {"Popularity": "popularity",
                       "Rating": "vote_weighted_average",
                       "Title": "original_title",
                       "Release Year": "release_date"}
    sorting_order = {"Ascending": True, "Descending": False}
    sort = sorting_methods[st.session_state.sorting_method]
    order = sorting_order[st.session_state.sorting_order]
    st.session_state.displayed_df.sort_values(
        by=[sort, "original_title"], ascending=[order, True], inplace=True)


sort_selection, sort_order, _ = st.columns([0.15, 0.15, 1])
with sort_selection:
    st.selectbox(label="Sorting method",
                 label_visibility="collapsed",
                 options=["Popularity", "Rating",
                          "Title", "Release Year"],
                 index=0,
                 on_change=sort_df,
                 key="sorting_method")
with sort_order:
    st.selectbox(label="Sorting order",
                 label_visibility="collapsed",
                 options=["Ascending", "Descending"],
                 index=1,
                 on_change=sort_df,
                 key="sorting_order")

sort_df()
st.session_state.displayed_df

st.write(f"## Movies By {st.session_state.sorting_method}")

# TODO: ADD ONLY THE NECESSARY COLUMNS BY LOOKING AT NUMBER OF ROWS IN DISPLAYED_DF
first_movie_row = st.columns(5)
second_movie_row = st.columns(5)
third_movie_row = st.columns(5)
fourth_movie_row = st.columns(5)
posters = [f"{idx}_w185.jpg" for idx in list(
    st.session_state.displayed_df['tmdbId'])]
titles = list(st.session_state.displayed_df['original_title'])
for i, col in enumerate(first_movie_row):
    try:
        col.image("assets/posters/" + posters[i], caption=f"{titles[i]}")
    except:
        col.image("assets/posters/null_w185.jpg", caption=f"{titles[i]}")
for i, col in enumerate(second_movie_row):
    try:
        col.image("assets/posters/" + posters[i+5], caption=f"{titles[i+5]}")
    except:
        col.image("assets/posters/null_w185.jpg", caption=f"{titles[i+5]}")
for i, col in enumerate(third_movie_row):
    try:
        col.image("assets/posters/" + posters[i+10], caption=f"{titles[i+10]}")
    except:
        col.image("assets/posters/null_w185.jpg", caption=f"{titles[i+10]}")
for i, col in enumerate(fourth_movie_row):
    try:
        col.image("assets/posters/" + posters[i+15], caption=f"{titles[i+15]}")
    except:
        col.image("assets/posters/null_w185.jpg", caption=f"{titles[i+15]}")


# TODO: NUMBER OF MOVIES NOT CORRECT WHEN FILTERS ARE REMOVED (NOT SURE WHEN RESET) - HAPPENS AT movies_idxs
# KINDA FIXED IT, NEED TO CHECK MORE DEEPLY - IT IS BECAUSE OF NULLS?? - CHECK DATA CLEANING

# TODO: FIX POSTERS NOT FOUND - DUE TO SEARCHING ONLY FOR "EN" TAG
#       ALSO OUT OF BOUNDS ERROR OFC - PROBABLY GONNA BE FIXED ABOVE WHEN ONLY DISPLAYING THE EXACT NUMBER OF COLUMNS/ROWS
#       AND SOME POSTERS HAVE DIFFERENT SIZES FOR SOME REASON??
