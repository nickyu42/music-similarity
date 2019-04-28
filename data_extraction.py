"""
Author: Nick Yu
Date created: 1/3/2019
"""
import pandas as pd
import pickle
import bs4
import requests
import time

import scraper

# Columns from MAL dataset to keep
COLUMNS = ['title', 'title_english', 'title_japanese', 'genre', 'opening_theme', 'ending_theme']

# sample count per label
# N shounen/ N shoujo
SAMPLE_SIZE = 1

WAIT_T = 0

MAL_PATH = 'data/AnimeList.csv'
PICKLE_PATH = 'data/animethemes_wiki.pickle'
CRED_PATH = 'data/credentials.json'
LOG_PATH = 'data/log.txt'
SAVE_PATH = 'data/songs/'

USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'


def filter_genre(genre):
    """
    Create filter function for row of pandas df
    :param genre: string genre to filter out
    :return: Function that returns True if genre is in the row genre string
    """
    def wrap(row):
        genres = row['genre']

        if isinstance(genres, str):
            return genre in genres.split(', ')

        return False

    return wrap


def find_samples(df, sample_size, existing):
    """
    Sample anime in df that have a link on /r/AnimeThemes
    :param df: dataframe to sample from
    :param sample_size: amount of samples to return
    :param existing: dict with downloadable links
    :return: list of samples
    """
    result = []

    count = 0
    while count < sample_size:
        row = df.sample()

        for t in ('title', 'title_english', 'title_japanese'):
            if row[t].values[0] in existing:
                result.append(row[t].values[0])
                count += 1
                break

    return result


# Read in the MAL dataset
mal_df = pd.read_csv(MAL_PATH, usecols=COLUMNS)

# Filter out all anime with 'Shounen', 'Shoujo' in the genre field
df_shounen = mal_df[mal_df.apply(filter_genre('Shounen'), axis=1)]
df_shoujo = mal_df[mal_df.apply(filter_genre('Shoujo'), axis=1)]

# Read in pickled html of the /r/AnimeThemes wiki
with open(PICKLE_PATH, 'rb') as f:
    wiki_raw = pickle.load(f)
    soup = bs4.BeautifulSoup(wiki_raw, 'html.parser')

# Load all links
wiki = soup.find(class_='md wiki')

existing_links = {}

# the first <p> is a message, so ignore
for link in wiki.find_all('p')[1:]:
    link = link.a

    # the names are in the format <name> (<year>)
    # remove the redundant year
    name = link.string.split('(')[0].strip()

    # add each link with the anime name as key
    # NOTE: doesn't matter if anime with duplicate name exist
    # as we don't care which version of op is used as long as both have same genres
    existing_links[name] = link.get('href')

shounen_samples = find_samples(df_shounen, SAMPLE_SIZE, existing_links)
shoujo_samples = find_samples(df_shoujo, SAMPLE_SIZE, existing_links)

reddit = scraper.create_praw(CRED_PATH)

cached_pages = {}

print('- Starting download -')

with open(LOG_PATH, 'w') as log:
    session = requests.Session()
    session.headers = {'User-Agent': USER_AGENT}

    for sample in shoujo_samples:
        year = existing_links[sample].split('/')[-1][:4]

        if year not in cached_pages:
            print(f'Grabbing links from {year}')
            cached_pages[year] = scraper.extract_songs(reddit, year)

        song_table = cached_pages[year]

        for song_type, link in song_table[sample]:

            # filter OP
            if song_type[:2] == 'OP':
                print(f'- {sample} : {song_type}')

                # download .webm
                response = session.get(link)

                video = response.content

                with open(f'{SAVE_PATH}/{sample}', 'wb') as wf:
                    wf.write(video)

                log.write(f'[SUCCESS] {sample} {song_type} {link}')
                break
        else:
            print(f'- No OP found for {sample}')
            log.write(f'[FAILED]  {sample}')

        if WAIT_T:
            time.sleep(WAIT_T)

