from sys import argv
from time import sleep

from duckduckgo_search import DDGS
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *


def search_images(keywords, max_images=200):
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')


def populate_image_directories(searches, path, sleep_time, mode):
    path = Path(path)
    max_images = 1 if mode == 'sample' else 200

    for o in searches:
        message = f'Adding {o} sample image' if mode == 'sample' else f'Populating {o} image directory'
        print(message)
        dest = (path/o)
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f'{o} photo', max_images))
        if mode == 'sample':
            continue

        sleep(sleep_time)
        download_images(dest, urls=search_images(f'{o} sun photo', max_images))
        sleep(sleep_time)
        download_images(dest, urls=search_images(f'{o} shade photo', max_images))
        resize_images(path/o, max_size=400, dest=path/o)

    return path  # For latter verification and DataBlock Construction


def failed_images_count(path):
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    return len(failed)


def learner_generator(keywords, path, sleep_time=10, export_name='temp_name'):
    print(f'Generating a learner to distiguish between {" and ".join(keywords)}')
    path = populate_image_directories(keywords, path, sleep_time, 'dev')
    print(f'Number of failed images: {failed_images_count(path)}')

    print('Populating DataBlock')
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path)

    print('Setting up Vision Learner')
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    print('Fine Tuning Learner')
    learn.fine_tune(3)
    print(f'Exporting learner model to {export_name}.pkl')
    learn.export(f'{export_name}.pkl')


def main():
    # User only provides keywords
    keywords = argv[1:]

    if not keywords:
        raise RuntimeError('No arguments provided: Please add desired image types you would like a trained model for')

    path = '_'.join(keywords)
    
    # populate a sample directory
    populate_image_directories(keywords, 'samples', 0, 'sample') 

    learner_generator(keywords, path, 10, path)
    print(f'{path} Model Generated.')


if __name__ == '__main__':
    main()
