# COL764 - Assignment 2: Document Reranking Task

## Description

The goal of this assignment is to develop "telescoping" models aimed at improving the precision of results using pseudo-relevance feedback. The dataset used for this assignment is the TREC COVID track. 

At the end, we need to re-rank the documents given to us using the various models which we implement (Rocchio, Lavrenko and Croft [RM1 and RM2]). The output also needs to be in the TREC format.

## Data description

* Document Collection (Contains the json files along with the metadata.csv which maps document IDs to their respective json (doc) file).
* Queries obtained from TREC COVID19 track.
* Pseudo relevance collection (Not given at the time of testing but to be used for tuning the hyperparameters). 
* Top100 documents given in TREC format (Needs to be re-ranked).

## Topics Covered

* Data analysis
* Pre-processing the data
* Rocchio's re-ranking algorithm
* Language Modelling introduction
* Lavrenko and Croft models (RM1 and RM2) along with Dirichlet smoothing.
* Hyperparameter Tuning
  
## Requirements

Python v3.7 was used for the purpose of this assignment. Moreover, the following libraries are also needed for the execution of these files along with the inbuilt python libraries:

* nltk
* bs4
* numpy
* pandas

Additionally, the code also requires `stopwords` corpora downloaded from NLTK. To download it, run the following in a python terminal:

```python 
import nltk
nltk.download("stopwords")
```


## Running the Code

* `rocchio_rerank.sh` - Reads the input files and reranks the given top100 results based on the *Rocchio's algorithm of Query Reformulation*
  
  `sh rocchio_rerank.sh <path_to_query_file> <path_to_top100_file> <collection_file> <output_file>`

    - This command will generate a single output file dictated by the `<output_file>` given as an argument. This file will contain reranked documents in the TREC eval format. Scores are also written along with the ranks.
  
* `lm_rerank.sh` - Reads the input files and reranks the given top100 results based on the *Lavrenko and Croft's* relevance model (both RM1 and RM2) with Dirichlet smoothing.
  
  `sh lm_rerank.sh <rm1|rm2> <path_to_query_file> <path_to_top100_file> <collection_file> <output_file> <expansions_file>`

    - This command will generate two output files directed by `<output_file>` and `<expansions_file>`. The output file will contain the reranked documents in TREC format and the expansions file will contain the 20 expansion words used for every query.
    - Specify the type of model to be used using rm1/rm2. rm1 is an indicative of the i.i.d. sampling model and rm2 is an indicative of the conditional sampling model.


## Alogrithmic Details and Analysis

Detailed analysis of the results has been done in the results pdf.