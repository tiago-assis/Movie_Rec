import requests
from PIL import Image
import pandas as pd
import time
import os
from dotenv import load_dotenv
from tqdm import tqdm
import json

HEADERS = {
    "accept": "application/json",
    "Authorization": os.getenv('TMDB_API_KEY')
}
POSTER_SIZE = "w342"
OVERWRITE_JSON = False

load_dotenv()

df = pd.read_csv("raw_data/movies_metadata.csv", usecols=['id'])
df.drop_duplicates(subset='id', inplace=True)

failed_image_urls = {}
failed_poster_urls = {}
saved_poster_urls = {}
for movie_id in tqdm(df["id"], total=len(df["id"]), mininterval=5):
    save_path = f"assets/posters/{movie_id}_{POSTER_SIZE}.jpg"
    if not os.path.exists(save_path):
        images_url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
        images_response = requests.get(images_url, headers=HEADERS)
        if images_response.status_code == 200:
            posters = images_response.json()["posters"]
            for poster in posters[::-1]:
                if poster['iso_639_1'] == "en":
                    poster_url = f"https://image.tmdb.org/t/p/{
                        POSTER_SIZE}{poster['file_path']}"
                    poster_response = requests.get(poster_url)
                    if poster_response.status_code == 200:
                        image = poster_response.content
                        with open(f"assets/posters/{movie_id}_{POSTER_SIZE}.jpg", "wb") as file:
                            file.write(image)
                        saved_poster_urls[movie_id] = poster_url
                    else:
                        print(f"Failed to get response from [{poster_url}]. Status Code {
                              poster_response.status_code}\n")
                        failed_poster_urls[movie_id] = poster_url
                break
        else:
            print(f"Failed to get response from [{images_url}]. Status Code {
                  images_response.status_code}\n")
            failed_image_urls[movie_id] = images_url

if not os.path.exists("saved_poster_urls.json") or OVERWRITE_JSON:
    with open("saved_poster_urls.json", "w") as f:
        json.dump(saved_poster_urls, f)
if not os.path.exists("failed_image_urls.json") or OVERWRITE_JSON:
    with open("failed_image_urls.json", "w") as f:
        json.dump(failed_image_urls, f)
if not os.path.exists("failed_poster_urls.json") or OVERWRITE_JSON:
    with open("failed_poster_urls.json", "w") as f:
        json.dump(failed_poster_urls, f)

# TODO: Comment code
