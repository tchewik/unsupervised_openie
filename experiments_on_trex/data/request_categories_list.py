import optparse

import requests
from tqdm import tqdm


def process_title(title):
    ## stopwords = [word.strip() for word in open('stopwords.filter', 'r').readlines()] # should not appear in the names
    stopwords = []

    for word in stopwords:
        if word in title.lower():
            return None
    return title


def main(input_file, output_file, max_depth=4):
    categories = []

    print(f"Extend categories from {input_file}.")
    with open(input_file, 'r') as f:
        max_depth, counter = max_depth, 0

        for category in tqdm(f.readlines()):

            params = {
                'categories': category,  # process the categories separately
                'depth': max_depth - min(counter, max_depth),
                'ns[14]': 1,  # namespace=14 is for categories, 0 for pages
                'language': 'en',
                'project': 'wikipedia',
                'format': 'json',
                'doit': 'Do it!'}

            r = requests.get('https://petscan.wmflabs.org/', params=params)
            data = r.json()

            for item in data['*'][0]['a']['*']:
                title = process_title(item.get('title'))
                if title and title not in categories:
                    categories.append(title)

    if categories:
        categories.sort()
        print(f"Extracted {len(categories)} subcategories from {categories[0]} to {categories[-1]}.\nWrite file {output_file}.")
        with open(output_file, 'w') as f:
            f.write('\n'.join(categories))


if __name__ == '__main__':
    parser = optparse.OptionParser(usage='Usage: %prog [options] [args]')

    parser.add_option('-i', dest='input_file', action='store', default='base_categories.txt', type='str',
                      help='input base categories list')
    parser.add_option('-o', dest='output_file', action='store', default='categories.txt', type='str',
                      help='extended wiki categories list')

    options, args = parser.parse_args()
    main(input_file=options.input_file, output_file=options.output_file)
