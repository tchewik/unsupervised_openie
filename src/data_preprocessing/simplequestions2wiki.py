import fire
import pandas as pd
from tqdm import tqdm
import json
import os
import bs4
import requests
import re

DEFAULT_DATA_PATH = '../../data/'
SIMPLEQUESTIONS_DIR = '../../data/simplequestions'
WIKITEXTS_DIR = '../../data/wiki_texts'
LINKS_FILENAME = 'simple_questions_wikipedia_links.json'

replace_map = {
    "\'s": "'s",
}

patterns = [re.compile(pat) for pat in [
    r'\[\d+\]',
    r'\xa0',
    r"\\'[^s]",
    r'\(.+\)'
]]


def clean_wiki_text(text):
    text = text.strip()
    while '\n\n' in text:
        text = text.replace('\n\n', '\n')

    for pattern in patterns:
        text = pattern.sub(' ', text)

    for pattern in replace_map:
        text = text.replace(pattern, replace_map.get(pattern))

    while '  ' in text:
        text = text.replace('  ', ' ')

    text = text.strip()
    if text[-1] == ':':
        text = text[:text.rfind('\n')]

    return text


def crawl_page(link):
    response = requests.get(link)

    if response is not None:
        html = bs4.BeautifulSoup(response.text, 'html.parser')

        title = html.select("#firstHeading")[0].text
        paragraphs = html.select("p")

        # just grab the text up to contents as stated in question
        text = '\n'.join([para.text.strip() for para in paragraphs])
        return clean_wiki_text(text)

    return None


def get_wikipedia_url_from_wikidata_id(wikidata_id, lang='en', debug=False):
    import requests
    from requests import utils

    url = (
        'https://www.wikidata.org/w/api.php'
        '?action=wbgetentities'
        '&props=sitelinks/urls'
        f'&ids={wikidata_id}'
        '&format=json')
    json_response = requests.get(url).json()
    if debug: print(wikidata_id, url, json_response)

    entities = json_response.get('entities')
    if entities:
        entity = entities.get(wikidata_id)
        if entity:
            sitelinks = entity.get('sitelinks')
            if sitelinks:
                if lang:
                    # filter only the specified language
                    sitelink = sitelinks.get(f'{lang}wiki')
                    if sitelink:
                        wiki_url = sitelink.get('url')
                        if wiki_url:
                            return requests.utils.unquote(wiki_url)
                else:
                    # return all of the urls
                    wiki_urls = {}
                    for key, sitelink in sitelinks.items():
                        wiki_url = sitelink.get('url')
                        if wiki_url:
                            wiki_urls[key] = requests.utils.unquote(wiki_url)
                    return wiki_urls
    return None


def sq2wiki(path=DEFAULT_DATA_PATH):
    """ Collects wikipedia pages from the path """

    if not os.path.isfile(os.path.join(path, LINKS_FILENAME)):
        print('Did not find the .json file with wikipedia links, make it from scratch...')

        data = {}
        for part in ["train", "valid", "test"]:
            path = os.path.join(SIMPLEQUESTIONS_DIR, f"annotated_wd_data_{part}_answerable_decoded.csv")
            data[part] = pd.read_csv(path)
        data = pd.concat([data[key] for key in data])

        subjectobject = data.subject_encoded.unique().tolist() + data.object_encoded.unique().tolist()
        subjectobject = list(set(subjectobject))

        links = {}
        for item in tqdm(subjectobject):
            new_link = get_wikipedia_url_from_wikidata_id(item)
            if new_link:
                links[item] = new_link

        with open(os.path.join(DEFAULT_DATA_PATH, LINKS_FILENAME), 'w') as f:
            json.dump(links, f)

    else:
        print('Found the .json file with wikipedia links, load it...')
        with open(os.path.join(path, LINKS_FILENAME), 'r') as f:
            links = json.load(f)

    print('Get the texts from Wikipedia...')
    for link in tqdm(links.keys()):
        if not os.path.isfile(os.path.join(WIKITEXTS_DIR, link + '.txt')):
            text = crawl_page(links.get(link))
            if text:
                with open(os.path.join(WIKITEXTS_DIR, link + '.txt'), 'w') as f:
                    f.write(text)


if __name__ == '__main__':
    fire.Fire(sq2wiki)
