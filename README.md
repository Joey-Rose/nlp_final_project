# Mitigating Gender Bias in Word Embeddings Via Corpus Augmentation and Vector Space Debiasing

## Project Overview
Multiple methods have been explored with the goal of mitigating gender bias in the domain of natural language processing. Two of the most prominent tackle the problem from different perspectives: debiasing training data and debiasing word embeddings. We took inspiration from [Bolukbasi et al.'s 2016 paper](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf) as well as from [Zhao et al.](https://arxiv.org/pdf/1804.06876.pdf)

The corpus, [LitBank](https://github.com/dbamman/litbank), is stored in the `novels` directory, with each novel saved as a .txt file. The `data` directory contains the lists of adjectives we use to evaluate the performance of the models, as well as dictionaries and lists from Bolukbasi containing gendered word-pairs that are used when calling their methods in `we.py` and `debias.py`. The base models are generated from `produce_models.py`, and are stored as binary files in the project root. In `test_models.py`, the binaries are loaded as Word2Vec objects. There is functionality in that file to create the word embedding lists from those models, but those embeddings are also stored as .txt files to save on compute time. Results generated from `test_models.py` are stored as CSV files in the project root, as well as further analysis files that have been done in Excel. 

## Running the project
We recommend first setting up a virtual environment through pip or through conda. Clone the project to your local machine, navigate to the project directory, and activate your environment. Once that is done, run:

`$ pip install -r requirements.txt` 

If that doesn't work, try:

`$ conda install -r requirements.txt`

You should now be able to run any of the files from the project. `test_models.py` is where a lot of the magic happens.

## Limitations
Some of the models do reduce the performance on useful gendered associations, something that can affect other downstream NLP tasks if one chooses to build off of our system. Secondly, two of the models make use of a technique that nearly doubles the corpus size. One of these that does not incorporate vector space debiasing does not show the most sizable improvement over the baseline model. Thus, if one chooses a larger corpus to train on, the gains against training time could be negligible or not worth it. 
