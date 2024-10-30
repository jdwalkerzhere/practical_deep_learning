from time import sleep

from duckduckgo_search import DDGS
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *


def search_images(keywords, max_images=200):
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')


def main():
    # Checking image search
    urls = search_images(keywords='bird photos', max_images=1)
    print(urls[0])


if __name__ == '__main__':
    main()
