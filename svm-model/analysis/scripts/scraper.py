"""
Author: Nick Yu
Date created: 1/3/2019

Module for scraping data from reddit
"""
import praw
import json
import bs4

USER_AGENT = 'AnimeThemes wiki scraper (by /u/nickyu42)'


def is_link(a):
    """Checks if given bs4 object is an <a> tag"""
    return isinstance(a, bs4.element.Tag) and a.has_attr('href')


def parse_row(row):
    """
    Parses an html table row where the first column is name
    and second column is a link
    :param row: bs4.element object
    :return: name, link pair
             e.g. ('OP', 'Crossing field')
    """
    cols = row.find_all(True)

    if is_link(cols[1].contents[0]):
        # name, link
        return cols[0].string, cols[1].a.get('href')

    return cols[0].string, None


def create_praw(cred_file):
    """Creates a Reddit object with credentials from given file"""
    with open(cred_file, 'r') as f:
        credentials = json.load(f)['praw']

    reddit = praw.Reddit(client_id=credentials['client_id'],
                         client_secret=credentials['client_secret'],
                         user_agent=USER_AGENT)

    return reddit


def get_wiki_links(wiki_page, year):
    """
    Extracts all headers and songs per header from a /r/AnimeThemes wiki page
    :param wiki_page: wikipage object
    :param year: which year of anime to scrape from the wiki
    :return: list of all headers and table objects in order
    """
    soup = bs4.BeautifulSoup(wiki_page[year].content_html, 'html.parser')
    wiki = soup.find(class_='md wiki')

    return wiki.find_all(['h3', 'table'])


def extract_songs(reddit, year):
    """
    Extracts links for all ops/eds within a single wiki page
    :param reddit: praw instance
    :param year: which wiki page year to scrape
    :return: dict with key as anime name and list of song, link pairs
    """
    wiki_page = reddit.subreddit('AnimeThemes').wiki
    links = get_wiki_links(wiki_page, year)

    song_table = {}
    current_anime = ''

    for l in links:
        if l.name == 'h3':
            current_anime = l.string

            # WARNING: duplicate detection is not done
            # another anime with the same name may break links
            song_table[current_anime] = []

        else:
            for row in l.tbody.find_all('tr'):
                name, link = parse_row(row)

                if link is not None:
                    song_table[current_anime].append((name, link))

    return song_table
