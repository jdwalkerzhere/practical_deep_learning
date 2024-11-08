# N-Keyword Image Classifier Generator
Having followed Chapter 1 of Practical Deep Learning, the aspect that stood out the most to me was the concept that a trained ML model is just another function that we can use on inputs down the line.

What I chose to do was make an extensible Generator for N-Keyword Image Classifiers.

By this I mean that if you want an image classifier that distiguishes coins, buttons, and bottlecaps, you can:
1. Clone this repo.
2. Activate a virtual environment
3. `pip3 install -r requirements.txt`
4. and run `python3 main.py coins buttons bottlecaps`

Running `main.py`:
1. Downloads some sample images for you to test your exported learner with later.
2. Populates the training data (this is the most time consuming step)
3. Build and pass the data blocks into a new learner
4. Fine tunes and exports that learner.

The generated fastai learner (for our hypothetical example above) will have the file name `coins_buttons_bottlecaps.pkl`.

This learner can subsequently be loaded into memory and used to predict other images.

I have also provided a simple (read as hacky and ugly) CLI based example (in `predict.py`) of using these exported learners that can be used to prove they work.
