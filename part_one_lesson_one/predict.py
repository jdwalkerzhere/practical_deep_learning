from os import listdir, path
from fastcore.all import *
from fastai.vision.all import *
from fastai.learner import load_learner
from PIL import Image


def main():
    selected_model = None
    while selected_model == None:
        present_models = [f for f in listdir('.') if path.isfile(f) and f.endswith('pkl')]
        present_models = {i: f for i, f in enumerate(present_models, start=1)}

        print("Which model do you want to use? Select with number.")
        for ind, model in present_models.items():
            print(f'{ind}. {model}')

        user_input = input()
        
        try:
            user_input = int(user_input) 
        except ValueError:
            print(f'Cannot convert {user_input} to integer. Please type a number')
            continue

        assert isinstance(user_input, int)

        selected_model = present_models.get(user_input, None)
        if not selected_model:
            print(f'Number {user_input} not present in the list. Please type a number that is present')
            continue

    print(f'Loading {selected_model} model')
    learner = load_learner(selected_model)

    image_path = input('What file do you want predictions for? Put in the path to the file\n')
    if not path.exists(image_path):
        raise RuntimeError(f'No path {image_path} exists')

    image = PILImage.create(image_path) 
    is_a, _, probs = learner.predict(image)
    print(f'This image is a {is_a}')
    print(f'Probability: {probs}')
        

if __name__ == '__main__':
    main()
