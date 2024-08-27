import streamlit as st
from st_click_detector import click_detector
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import pickle
import os
import math
import time
import ast

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
    with open("cleaned_data/recommendation_movie_mappings.pickle", "rb") as f:
        rec_movie_mappings = pickle.load(f)
    movie_recs = load_npz("cleaned_data/movie_recommendations.npz")
    return df, genres_cols, spoken_languages_cols, cast_cols, production_companies_cols, movie_recs, rec_movie_mappings


df, genres_cols, spoken_languages_cols, cast_cols, production_companies_cols, movie_recs, rec_movie_mappings = get_data()


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

sidebar = st.sidebar
with sidebar:
    st.markdown("""
            <style>
            p {
                text-align: justify;
            }
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
                   options=dict(sorted(st.session_state.movie_label_counter['production_companies'].items(
                   ), key=lambda counter: counter[1], reverse=True)),
                   format_func=lambda production_company: f"""{production_company} ({
                       st.session_state.movie_label_counter['production_companies'][production_company]})""",
                   default=st.session_state.production_company_selection if "production_company_selection" in st.session_state else None,
                   on_change=filter_by_production_company, key="production_company_selection",
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
    sorting_selection_options = [
        "Popularity", "Rating", "Title", "Release Year"]
    st.selectbox(label="Sorting method",
                 label_visibility="collapsed",
                 options=sorting_selection_options,
                 index=sorting_selection_options.index(
                     st.session_state.sorting_method) if "sorting_method" in st.session_state else 0,
                 on_change=sort_df,
                 key="sorting_method")
with sort_order:
    sorting_order_options = ["Ascending", "Descending"]
    st.selectbox(label="Sorting order",
                 label_visibility="collapsed",
                 options=sorting_order_options,
                 index=sorting_order_options.index(
                     st.session_state.sorting_order) if "sorting_order" in st.session_state else 1,
                 on_change=sort_df,
                 key="sorting_order")

sort_df()
st.session_state.displayed_df

st.write(f"## Movies By {st.session_state.sorting_method}")

#! DON'T KNOW IF THE FOLLOWING WORKS FOR ANY VALUE OF MAX_ROWS AND MAX_COLUMNS !#
# MAX_ROWS = 4
# MAX_COLUMNS = 5
# total_filtered_movies = len(st.session_state.displayed_df)
# row_calc = total_filtered_movies / MAX_COLUMNS
# number_of_rows = min(MAX_ROWS, math.ceil(row_calc))
# number_of_columns_in_last_row = math.ceil(
#    total_filtered_movies % MAX_COLUMNS) or MAX_COLUMNS
# columns_per_row = [MAX_COLUMNS] * \
#    (number_of_rows - 1) + [number_of_columns_in_last_row]
# poster_columns = [st.columns(MAX_COLUMNS)] * number_of_rows
#
# posters = [f"{idx}_w500.jpg" for idx in list(
#    st.session_state.displayed_df['tmdbId'])]
titles = list(st.session_state.displayed_df['original_title'])

html_posters = [f"""<td><a href='#' id='{idx}'><img width='93%' src='https://raw.githubusercontent.com/tiago-assis/Movie_Recommendation/main/assets/posters/{
    idx}_w500.jpg'></a></td>""" for idx in list(st.session_state.displayed_df['tmdbId'])]

MAX_COLUMNS = 5
MAX_ROWS = 4
start_display_index = 0
end_display_index = MAX_ROWS * MAX_COLUMNS
html_posters_to_display = "<table>"
for i in range(start_display_index, end_display_index, MAX_COLUMNS):
    html_posters_to_display += f"""<tr>{
        "\n".join(html_posters[i:i+MAX_COLUMNS])}</tr><tr><td><p></p></td></tr>"""
html_posters_to_display += "</table>"

# for i, row_columns in enumerate(poster_columns):
#    for j, col in enumerate(row_columns[:columns_per_row[i]]):
#        try:
#            col.image("assets/posters/" +
#                      posters[i * MAX_COLUMNS + j], caption=f"{titles[i * MAX_COLUMNS + j]}")
#        except:
#            col.image("assets/posters/null_w500.jpg",
#                      caption=f"{titles[i * MAX_COLUMNS + j]}")


def get_recommendations(movie_id, knn=5):
    idx = df.loc[df['tmdbId'] == movie_id, "movieId"].values[0]
    mapped_idx = rec_movie_mappings[idx]
    sims = movie_recs.getrow(mapped_idx).todense()
    sims = sims.argsort().A1[:-(knn+2):-1][1:]
    reverse_rec_movie_mappings = {v: k for k, v in rec_movie_mappings.items()}
    sim_idxs = []
    # TODO: PARALLELIZE THIS MAYBE
    for idx in sims:
        sim_idxs.append(reverse_rec_movie_mappings.get(idx))
    return df.loc[df['movieId'].isin(sim_idxs), "tmdbId"]


@st.dialog("Movie Details", width='large')
def display_movie(movie_id):
    movie_img, movie_info = st.columns([1.5, 2.5], gap='medium')
    with movie_img:
        st.image(f"assets/posters/{movie_id}_w500.jpg")
    with movie_info:
        filter = df['tmdbId'] == movie_id
        title = df.loc[filter, 'original_title'].values[0]
        year = df.loc[filter, 'release_date'].values[0]
        rating = df.loc[filter, 'vote_weighted_average'].values[0]
        overview = df.loc[filter, 'overview'].values[0]
        genres = ast.literal_eval(df.loc[filter, 'genres'].values[0])
        prod_companies = ast.literal_eval(
            df.loc[filter, 'production_companies'].values[0])
        imdb_page = f"https://www.imdb.com/title/{
            df.loc[filter, 'imdb_id'].values[0]}"
        homepage = df.loc[filter, 'homepage'].values[0]

        movie_title, movie_rating = st.columns([0.8, 0.2])
        with movie_title:
            st.write(f"## {title} ({year})")
        with movie_rating:
            st.markdown(
                f"""<h2 style='text-align:right'>{round(rating, 1)} ⭐</h2>""", unsafe_allow_html=True)
        st.write(f"{overview}")
        st.write(f"**Genres:** {', '.join(genres)}")
        st.markdown(f"**Production Companies:** {', '.join(prod_companies)}")
        st.markdown(f"""**More Info:** [IMDB Page]({imdb_page}){
                    f', [Movie Homepage]({homepage})' if not isinstance(homepage, float) else ''}""")

    st.write("**Movie Recomendations:**")
    movie_rec_ids = get_recommendations(movie_id)
    recs = st.columns(len(movie_rec_ids))
    for i, id in enumerate(movie_rec_ids):
        recs[i].image(f"assets/posters/{id}_w500.jpg")


clicked = click_detector(html_posters_to_display)
if clicked != "":
    display_movie(int(clicked))


# TODO: ABLE TO CLICK RECOMMENDED MOVIES
# TODO: MORE ROBUST RECOMMENDATIONS
# TODO: reset button can be spammed and errors occur due to missing session_state keys
####!!!! TODO: ACRESCENTAR O RESTO DAS MATRIZES AO CACHE !!!!####
