import csv
import re
from bs4 import BeautifulSoup

# curl --silent "https://old.reddit.com/r/AnimeThemes/wiki/anime_index" > wiki.html

with open('wiki.html', 'r') as f:
    t = f.read()

soup = BeautifulSoup(t, features='html.parser')

with open('out.csv', 'w') as f:
    writer = csv.writer(f)

    for link in soup.select('.md.wiki a'):
        if not link.has_attr('rel'):
            continue
        
        writer.writerow([link.text, link.attrs['href']])
