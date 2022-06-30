# Using Language Models for Analyzing Semantic Variation between Dutch Social Communities

This is the repository for a thesis submitted in partial fulfilment of the requirements for the degree of MA Linguistics (Human Language Technology) at VU University Amsterdam.
June 2022.

### Project

This project aims to investigate semantic variation in social settings using language models.

## Getting started

### Requirements

This codebase is written entirely in Python 3.7. requirements.txt contains all necessary packages to run the code successfully. These are easy to install via pip using the following instruction:

```
pip install -r requirements.txt
```

### Structure

The following list describes the folders in the project:

- **/code**: contains all code of this project
- **/data**: contains data used for this project

## Using the main programs

Two main components of this project concern 1) the extraction of word representations from the data and 2) the measurement of distance between them. 

1. **Representations**
Change the working directory of the execution environment to the **/code/representations** directory.
Extracting PPMI-vectors for a set of target words in a dataset can be run using the following instruction:
  ```
    python ppmi.py --output_dir [DIRECTORY] --targets_path [FILEPATH] --dataset_path [FILEPATH] --window_size [WINDOW SIZE] --alpha [ALPHA] --k [K] 
  ```
Extracting contexualized representations for a set of target words in a dataset can be run using the following instruction:
  ```
    python bertje_embeddings.py --model_dir [DIRECTORY] --output_path [FILEPATH] --targets_path [FILEPATH]  --dataset_path [FILEPATH] --batch_size [BATCH SIZE] --pretrained [TRUE/FALSE] --lemma [TRUE/FALSE] --layers [ALL/TOP]
  ```

2. **Distance**
Change the working directory of the execution environment to the **/code/distance** directory. 
Measuring the cosine distance between PPMI-vectors can be run using the following instruction: 
  ```
    python cosine_distance.py --targets_path [FILEPATH] --ppmi_path1 [FILEPATH] --ppmi_path2 [FILEPATH] --vocab_path1 [FILEPATH] --vocab_path2 [FILEPATH] --output_path 
  ```
Measuring the average pairwise distance between contextualized representations can be run using the following instruction: 
  ```
    python average_pairwise_distance.py --targets_path [FILEPATH] --usage_path1 [FILEPATH] --usage_path2 [FILEPATH] --output_path [FILEPATH]
  ```


## Author
- Sanne Hoeken (student number: 2710599)
