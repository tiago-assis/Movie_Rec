import requests
import pandas as pd
import os
from dotenv import load_dotenv
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

HEADERS = {
    "accept": "application/json",
    "Authorization": os.getenv('TMDB_API_KEY')
}
POSTER_SIZE = "w500"
SAVE_DIR = "assets/posters"
OVERWRITE_JSON = True

load_dotenv()

df = pd.read_csv(
    "cleaned_data/movies_metadata_small_cleaned.csv", usecols=['tmdbId', 'original_language'])
df.drop_duplicates(subset='tmdbId', inplace=True)

failed_access_urls = {}
failed_poster_urls = {}
saved_poster_urls = {}


def get_poster_url(movie_id, original_language, all_languages=False):
    base_url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
    if all_languages:
        return base_url
    return f"{base_url}?include_image_language={original_language}"


def save_poster(image_content, save_path):
    with open(save_path, "wb") as file:
        file.write(image_content)


def fetch_and_save_poster(movie_id, poster_url, save_path):
    poster_response = requests.get(poster_url)
    if poster_response.status_code == 200:
        save_poster(poster_response.content, save_path)
        saved_poster_urls[movie_id] = poster_url
    else:
        failed_poster_urls[movie_id] = poster_url
        print(f"""\nFailed to get response from [{poster_url}]. Status Code {
            poster_response.status_code}""")


def main(row):
    movie_id = row['tmdbId']
    original_language = row['original_language']
    save_path = f"{SAVE_DIR}/{movie_id}_{POSTER_SIZE}.jpg"
    if os.path.exists(save_path):
        return

    for all_languages in [False, True]:
        images_url = get_poster_url(movie_id, original_language, all_languages)
        images_response = requests.get(images_url, headers=HEADERS)

        if images_response.status_code == 200:
            posters = images_response.json().get("posters", [])
            if posters:
                poster_url = f"""https://image.tmdb.org/t/p/{
                    POSTER_SIZE}{posters[0]['file_path']}"""
                fetch_and_save_poster(movie_id, poster_url, save_path)
                return
            elif all_languages:
                failed_access_urls[movie_id] = images_url
                print(f"""\nFailed to get response from [{images_url}]. Status Code {
                    images_response.status_code}""")


with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 5) as executor:
    futures = [executor.submit(main, row) for _, row in df.iterrows()]

    for future in tqdm(as_completed(futures), total=len(futures), mininterval=3):
        future.result()

# df.apply(get_posters, axis=1)


# def get_posters(row, all_languages=False, fail_flag=False):
#    t.update(1)
#    movie_id = row['tmdbId']
#    original_language = row['original_language']
#    save_path = f"{SAVE_DIR}/{movie_id}_{POSTER_SIZE}.jpg"
#    if all_languages:
#        images_url = f"""https://api.themoviedb.org/3/movie/{
#            movie_id}/images"""
#    else:
#        images_url = f"""https://api.themoviedb.org/3/movie/{
#            movie_id}/images?include_image_language={original_language}"""
#    images_response = requests.get(images_url, headers=HEADERS)
#    if images_response.status_code == 200:
#        posters = images_response.json()["posters"]
#        if len(posters) > 0:
#            poster_url = f"""https://image.tmdb.org/t/p/{
#                POSTER_SIZE}{posters[0]['file_path']}"""
#            poster_response = requests.get(poster_url)
#            if poster_response.status_code == 200:
#                image = poster_response.content
#                with open(save_path, "wb") as file:
#                    file.write(image)
#                saved_poster_urls[movie_id] = poster_url
#            else:
#                print(f"""Failed to get response from [{poster_url}]. Status Code {
#                    poster_response.status_code}\n""")
#                failed_poster_urls[movie_id] = poster_url
#        else:
#            if fail_flag:
#                print(f"""Failed to get posters from [{
#                      images_url}] although a successfull response was received.""")
#                failed_access_urls[movie_id] = images_url
#                return
#            get_posters(row, all_languages=True, fail_flag=True)
#    else:
#        print(f"""Failed to get response from [{images_url}]. Status Code {
#              images_response.status_code}\n""")
#        failed_access_urls[movie_id] = images_url


# for movie_id, original_language in tqdm(df["tmdbId", "original_language"], total=len(df), mininterval=5):
#    get_poster(movie_id, original_language)

# for movie_id in tqdm(df["tmdbId"], total=len(df["tmdbId"]), mininterval=5):
#    save_path = f"{SAVE_DIR}/{movie_id}_{POSTER_SIZE}.jpg"
#    images_url = f"https://api.themoviedb.org/3/movie/{
#        movie_id}/images?include_image_language=en"
#    images_response = requests.get(images_url, headers=HEADERS)
#    if images_response.status_code == 200:
#        try:
#            posters = images_response.json()["posters"]
#            poster_url = f"https://image.tmdb.org/t/p/{
#                POSTER_SIZE}{posters[-1]['file_path']}"
#            poster_response = requests.get(poster_url)
#            if poster_response.status_code == 200:
#                image = poster_response.content
#                with open(save_path, "wb") as file:
#                    file.write(image)
#                saved_poster_urls[movie_id] = poster_url
#            else:
#                print(f"""Failed to get response from [{poster_url}]. Status Code {
#                    poster_response.status_code}\n""")
#                failed_poster_urls[movie_id] = poster_url
#        except:
#            print(f"""Failed to get posters from [{
#                  images_url}] with successful server response\n""")
#            failed_access_urls[movie_id] = images_url
#    else:
#        print(f"""Failed to get response from [{images_url}]. Status Code {
#              images_response.status_code}\n""")
#        failed_access_urls[movie_id] = images_url


if not os.path.exists("assets/saved_poster_urls.json") or OVERWRITE_JSON:
    with open("assets/saved_poster_urls.json", "w") as f:
        json.dump(saved_poster_urls, f)
if not os.path.exists("assets/failed_access_urls.json") or OVERWRITE_JSON:
    with open("assets/failed_access_urls.json", "w") as f:
        json.dump(failed_access_urls, f)
if not os.path.exists("assets/failed_poster_urls.json") or OVERWRITE_JSON:
    with open("assets/failed_poster_urls.json", "w") as f:
        json.dump(failed_poster_urls, f)


# TODO: Comment code
