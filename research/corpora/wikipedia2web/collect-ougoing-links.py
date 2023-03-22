from argparse import ArgumentParser
from pprint import pprint

import requests
import wikipediaapi


wiki_wiki = wikipediaapi.Wikipedia(
    language='uk',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('output', help='txt file with each line containing new link')
    args = parser.parse_args()

    S = requests.Session()

    URL = "https://en.wikipedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "titles": "Україна",
        "prop": "extlinks",
        "format": "json"
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    print(DATA)
