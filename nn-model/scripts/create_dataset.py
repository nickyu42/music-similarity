import os
import re
import csv

import bs4
import praw
import dotenv
import pandas as pd
import string_grouper

dotenv.load_dotenv()

pattern = re.compile(r'\(\d{4}\)')


def remove_year(title):
    return pattern.sub('', title).strip()


def get_year(link):
    l = link.lstrip('/r/AnimeThemes/wiki/').split('#')
    if len(l) == 1:
        return None
    return l[0]


def get_id(link):
    l = link.lstrip('/r/AnimeThemes/wiki/').split('#')
    if len(l) == 1:
        return None
    return l[1]


corpus = pd.read_csv('animes.csv')
to_match = pd.read_csv('out.csv', names=('name', 'link'))
to_match.set_index('name')

to_match['name'] = to_match['name'].map(remove_year)
to_match['year'] = to_match['link'].map(get_year)
to_match['id'] = to_match['link'].map(get_id)

sh = corpus[corpus['genre'].str.contains('Shounen', regex=False)]
sj = corpus[corpus['genre'].str.contains('Shoujo', regex=False)]

sim_threshold = 0.85

sh_matches = string_grouper.match_strings(sh['title'], to_match['name'])
sh_indices = pd.unique(
    sh_matches[sh_matches['similarity'] > sim_threshold]['right_index'])
sh_matches = to_match.iloc[sh_indices]

sj_matches = string_grouper.match_strings(sj['title'], to_match['name'])
sj_indices = pd.unique(
    sj_matches[sj_matches['similarity'] > sim_threshold]['right_index'])
sj_matches = to_match.iloc[sj_indices]

print(f'{len(sh_indices)=} {len(sj_indices)=}')


def get_links(df: pd.DataFrame, wiki, cached_pages, filename):

    with open('op_' + filename, 'w') as op_f, open('ed_' + filename, 'w') as ed_f:
        op_w = csv.writer(op_f)
        op_w.writerow(['name', 'link'])

        ed_w = csv.writer(ed_f)
        ed_w.writerow(['name', 'link'])

        for _, row in df.iterrows():
            year = row['year']
            id_ = row['id']
            series_name = row['name']

            print(series_name)

            if year in cached_pages:
                print(f'\t Using cached {year}')
                page: bs4.BeautifulSoup = cached_pages[year]
            else:
                print(f'[INFO] Getting {year}')
                try:
                    html = wiki[year].content_html
                except Exception as e:
                    print(f'[ERROR] {series_name} : {e}')
                    continue

                print(f'\t - Got raw HTML')
                page = bs4.BeautifulSoup(html, features='html.parser')
                print(f'\t - Parsing')
                cached_pages[year] = page.find(class_='md wiki')
                print(f'\t Done')

            header = page.find(id=id_)
            
            if header is None:
                print(f'[ERROR] {series_name} is missing')
                continue

            table = header.find_next('table').select_one('tbody')

            for tr in table.find_all('tr'):
                data = tr.find_all('td')
                name = data[0].string
                link = data[1].find('a')

                if link is None:
                    continue
                else:
                    link = link.attrs['href']

                if name is None:
                    continue

                if 'OP' in name:
                    op_w.writerow((series_name, link))
                    op_f.flush()

                if 'ED' in name:
                    ed_w.writerow((series_name, link))
                    ed_f.flush()


user_agent = 'AnimeThemes wiki scraper (by /u/nickyu42)'
reddit = praw.Reddit(client_id=os.environ['REDDIT_CLIENT_ID'],
                     client_secret=os.environ['REDDIT_CLIENT_SECRET'],
                     user_agent=user_agent)

wiki = reddit.subreddit('AnimeThemes').wiki

cached_pages = {}
get_links(sh_matches, wiki, cached_pages, 'sh.csv')
get_links(sj_matches, wiki, cached_pages, 'sj.csv')
