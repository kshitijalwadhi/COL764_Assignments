# coding: utf-8
from nltk.corpus import stopwords
from tqdm import tqdm
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import nltk
import re
import os
import math
import json
import sys
import time
import json
from PorterStemmer import *
#from nltk.stem import PorterStemmer
#from nltk.tokenize import word_tokenize
ps = PorterStemmer()

stop_words = set(stopwords.words('english'))

delim = '''[ ',(){}.:;"`\n]'''


data_dir = "C:\Files\col764-a2-release\col764-ass2-release"


metadata = os.path.join(data_dir, "metadata.csv")

meta_df = pd.read_csv(metadata)
meta_df = meta_df[['cord_uid', 'title', 'abstract', 'pdf_json_files', 'pmc_json_files']]
meta_df['pdf_json_files'] = meta_df['pdf_json_files'].astype(str)
meta_df['pmc_json_files'] = meta_df['pmc_json_files'].astype(str)


def getTokensFromText(text):
    words = re.split(delim, text)
    res = []
    for w in words:
        if(len(w) > 0):
            #temp = ps.stem(w.lower(), 0, len(w)-1)
            temp = w.lower()
            if temp not in stop_words:
                res.append(temp)
    return res


def get_idf_dict_new():
    idf_dict = {}
    filemap = {}
    for row in tqdm(meta_df.itertuples(), total=meta_df.shape[0]):
        uid = row.cord_uid
        alltext = ""
        if row.pmc_json_files:
            paths = row.pmc_json_files
        elif row.pdf_json_files:
            paths = row.pdf_json_files
        filemap[uid] = paths
        for path in paths.split("; "):
            if(path == "nan"):
                continue
            datapath = os.path.join(data_dir, path)
            with open(datapath) as data:
                data_dict = json.load(data)
                for section in data_dict['body_text']:
                    section_text = section['text']
                    alltext += section_text
                    alltext += " "
        tokens = getTokensFromText(alltext)
        tokens = set(tokens)
        for token in tokens:
            if token not in idf_dict:
                idf_dict[token] = 1
            else:
                idf_dict[token] += 1
    return idf_dict, filemap


idf_dict, filemap = get_idf_dict_new()


def dumpdicts(idf_dict, filemap):
    with open("idf_dict.json", "w") as outfile:
        json.dump(idf_dict, outfile)
    with open("filemap.json", "w") as outfile:
        json.dump(filemap, outfile)


dumpdicts(idf_dict, filemap)
