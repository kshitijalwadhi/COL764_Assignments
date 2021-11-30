# COL764 - Assignment 1: Inverted Index

## Description
In this assignment, we implemented a Boolean Retrieval Model for Information Retrieval. An efficient-to-query Inverted Index was created on-disk to save RAM usage.

## Topics Covered

* Pre-Processing the Data
* Creating Inverted Index
* Dumping this inverted index into a temporary file
* Merging the inverted index files created
* Encoding
* Retrieval
* Performance

## Pre Steps

* pip install beautifulsoup4
* pip install python-snappy

## Submission files

The submission contains 2 shell files which call the respective python files. 

* `invidx_cons.sh` - Reads the input files, preprocesses the data and generates the posting list.

  `python invidx_cons.py <path to data> <name of index file to generate> <stopwords file> <compression type> <xmltags path>`

  - This command will also generate temporary files in the same directory where the shell file is present but they will be deleted at the end of the program. 

  - At the end, this program will generate two files, a .dict file and a .idx file which contains the mappings and the postings list repectively.
  
* `boolsearch.sh` - Performs boolean retrieval on the queries provided to us using the inverted index we created before.
  
    `python boolsearch.py <query file> <name of result file to generate> <path to index file> <path to dict file>`

    - This program will generate the result of the queries in the file given as arguments. 
  
Along with these, the directory also contains the PorterStemmer.py file.


## Running the code

* `bash invidx_cons.sh <path to data> <name of index file to generate> <stopwords file> <compression type> <xmltags path>`
* `bash boolsearch.sh <query file> <name of result file to generate> <path to index file> <path to dict file>`