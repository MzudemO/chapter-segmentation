# Code for the Master's Thesis "Chapter Segmentation in German Literary Texts"

This repository contains the code used to create the dataset for chapter segmentation.
Code used for experiments remains for archival purposes and was not made to work OOB.

This dataset was created using the [Gutenberg-DE Edition 16 digital download](https://shop.projekt-gutenberg.org/home/446-gutenberg-de-editon-16-zum-download-9783739011707.html).
The dataset parser expects the archive extracted on disk.
Other editions were not tested.

To create the dataset from the thesis:

```sh
# install required python packages for dataset parsing
# the use of a virtual environment is recommended
pip install -r requirements.txt
# create the dataset in the corpus folder
# see python parse.py -h for additional arguments
# some decode errors are expected
python parse.py <path-to>/gutenberg-edition16/
```

From then on, it is possible to recreate experiments, though this will likely require editing some of the code and installing additional dependencies not included in the `requirements.txt`.

```sh
# create tokenized, labelled sequence pairs for BERT
# NOTE: creates ~21GB of csv, long running time  
python onetime_scripts/transformer_dataset.py
# undersample training dataset if necessary (paper uses 1:1)
python onetime_scripts/balanced_dataset.py

# adjust pytorch_trainer.py to read the dataset from disk
# auth key is not needed unless using data from huggingface hub
# run training
python pytorch_trainer.py

# generate predictions on test set
python get_model_predictions.py
# methods for evaluation are in onetime_scripts/evaluate_predictions.ipynb 
```

