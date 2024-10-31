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
    dest = 'bird.jpg'
    download_url(urls[0], dest, show_progress=False)
    im = Image.open(dest)
    im.thumbnail(256,256).save(dest)
    download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
    Image.open('forest.jpg').to_thumb(256,256)


if __name__ == '__main__':
    main()
