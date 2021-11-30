# coding: utf-8
from nltk.corpus import stopwords
import random
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
import re
import csv
from PorterStemmer import *
ps = PorterStemmer()
random.seed(42)


stop_words = set(stopwords.words('english'))
delim = '''[ ',(){}.:;"`\n]'''


def getTokensFromText(text):
    words = re.split(delim, text)
    res = []
    for w in words:
        if(len(w) > 2):
            #temp = ps.stem(w.lower(), 0, len(w)-1)
            temp = w.lower()
            if temp not in stop_words and not re.search('[0-9]+', temp):
                res.append(temp)
    return res


def get_idf_dict_new():
    idf_dict = {}
    filemap = {}

    with open(metadata, 'r', encoding='utf-8') as metafile:
        reader = csv.DictReader(metafile)
        for row in reader:
            uid = row['cord_uid']
            paths = ""
            alltext = ""
            if row['pmc_json_files']:
                paths = row['pmc_json_files']
            elif row['pdf_json_files']:
                paths = row['pdf_json_files']
            filemap[uid] = paths
            if paths != "":
                for path in paths.split("; "):
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


def dumpdicts(idf_dict, filemap):
    with open("idf_dict.json", "w") as outfile:
        json.dump(idf_dict, outfile)
    with open("filemap.json", "w") as outfile:
        json.dump(filemap, outfile)


def create_vocab_dict(idf_dict):
    vocab = {}
    for i, word in enumerate(idf_dict.keys()):
        vocab[word] = i
    return vocab


def get_tf_dict(docid):
    tf_dict = {}
    paths = filemap[docid]
    alltext = ""
    if paths != "":
        for path in paths.split("; "):
            datapath = os.path.join(data_dir, path)
            with open(datapath) as data:
                data_dict = json.load(data)
                for section in data_dict['body_text']:
                    section_text = section['text']
                    alltext += section_text
                    alltext += " "
    tokens = getTokensFromText(alltext)
    for token in tokens:
        if token not in tf_dict:
            tf_dict[token] = 1
        else:
            tf_dict[token] += 1
    return tf_dict


def get_idf(term):
    return math.log(1+N/idf_dict[term], 2)


def getDocVector(docid):
    tf_dict = get_tf_dict(docid)
    vec = np.zeros(len(vocab))
    for word, tf in tf_dict.items():
        if word in vocab:
            pos = vocab[word]
            normtf = 1 + math.log(tf, 2)
            normidf = get_idf(word)
            tfidf = normtf * normidf
            vec[pos] = tfidf
    return vec


def getQueryVector(query):
    tokens = getTokensFromText(query)
    query_tf_dict = {}
    for token in tokens:
        if token not in query_tf_dict:
            query_tf_dict[token] = 1
        else:
            query_tf_dict[token] += 1
    vec = np.zeros(len(vocab))
    for word, tf in query_tf_dict.items():
        if word in vocab.keys():
            pos = vocab[word]
            normtf = 1 + math.log(tf, 2)
            normidf = get_idf(word)
            tfidf = normtf * normidf
            vec[pos] = tfidf
    return vec


def getSim(queryVec, docVec):
    if np.linalg.norm(queryVec) == 0 or np.linalg.norm(docVec) == 0:
        return 0.0
    return np.dot(queryVec, docVec)/(np.linalg.norm(queryVec) * np.linalg.norm(docVec))


def get_queries():
    global offset
    with open(os.path.join(data_dir, topics_file), "r") as file:
        content = file.readlines()
        content = "".join(content)
        bs_content = BeautifulSoup(content, "lxml")
        offset = bs_content.find('topic', {'number': True}).attrs['number']
        offset = int(offset)
        queries = bs_content.find_all("query")
        all_text = []
        for q in queries:
            all_text.append(q.text)
        return all_text


def get_top100():
    top100_dict = {}
    with open(os.path.join(data_dir, top100_file), "r") as file:
        for x in file:
            line = x.split(" ")
            if(len(line) > 1):
                qid = int(line[0])
                docid = line[2]
                if qid not in top100_dict:
                    top100_dict[qid] = [docid]
                else:
                    top100_dict[qid].append(docid)
    return top100_dict


def getNDCG(qno, docs):
    def discount_helper(arr):
        score = 0
        for i, e in enumerate(arr):
            score += e/math.log(i+2, 2)
        return score
    scores = []
    for doc in docs:
        if doc in rel_dict[qno]:
            scores.append(rel_dict[qno][doc])
        else:
            scores.append(0)
    origscore = discount_helper(scores)
    sortedscore = discount_helper(sorted(scores, reverse=True))
    if sortedscore == 0:
        return 0
    return origscore/sortedscore


def getAvgNDCG():
    total = 0
    all_scores = []
    for i in range(len(top100_dict)):
        qno = i+offset
        qndcg = getNDCG(qno, top100_dict[qno])
        all_scores.append(qndcg)
        total += qndcg
    return all_scores, total/len(top100_dict)


def getRandomDocs(num):
    elems = random.sample(list(filemap), num)
    return elems


def getIrrelVec():
    irrelvec = np.zeros(len(vocab))
    randomdocs = getRandomDocs(100)
    irrel_vecarr = []
    for doc in randomdocs:
        irrel_vecarr.append(getDocVector(doc))
    for v in irrel_vecarr:
        irrelvec += v
    if len(irrel_vecarr) > 0:
        irrelvec /= len(irrel_vecarr)
    return irrelvec


def rocchio_updateQueriesTest(alpha, beta, gamma):
    updated_queries = []
    for i, q_orig in (enumerate(get_queries())):
        q = q_orig.lower()
        qv0 = getQueryVector(q)
        qno = i+offset
        top100docs = top100_dict[qno]
        relvec = np.zeros(len(vocab))
        rel_vecarr = []
        for doc in top100docs:
            rel_vecarr.append(getDocVector(doc))
        for v in rel_vecarr:
            relvec += v
        if len(rel_vecarr) > 0:
            relvec /= len(rel_vecarr)
        qnew = np.multiply(qv0, alpha) + np.multiply(relvec, beta) - np.multiply(irrelvec, gamma)
        qnew = np.where(qnew < 0, 0, qnew)
        updated_queries.append(qnew)
    return updated_queries


def rocchio_getNDCG(updated_queries):
    total = 0
    allscores = []
    for i, q in (enumerate(updated_queries)):
        qno = i+offset
        doc_score_dict = {}
        top100docs = top100_dict[qno]
        for doc in top100docs:
            score = getSim(q, getDocVector(doc))
            doc_score_dict[doc] = score
        sorted_docs = dict(sorted(doc_score_dict.items(), key=lambda x: x[1], reverse=True))
        sorted_docs = list(sorted_docs.keys())
        ndcg = getNDCG(qno, sorted_docs)
        allscores.append(ndcg)
        total += ndcg
    return allscores, total/len(updated_queries)


def getUpdatedOrder(updated_queries):
    res_file = open(output_file, "w")
    for i, q in (enumerate(updated_queries)):
        qno = i+offset
        doc_score_dict = {}
        top100docs = top100_dict[qno]
        for doc in top100docs:
            score = getSim(q, getDocVector(doc))
            doc_score_dict[doc] = score
        sorted_docs = dict(sorted(doc_score_dict.items(), key=lambda x: x[1], reverse=True))
        for r, doc in enumerate(sorted_docs.keys()):
            rank = r+1
            score = sorted_docs[doc]
            l = str(qno) + " Q0 " + doc + " " + str(rank) + " " + str(score) + " " + "runid1\n"
            res_file.write(l)
    res_file.close()


offset = 1

if __name__ == "__main__":

    topics_file = sys.argv[1]
    top100_file = sys.argv[2]
    data_dir = sys.argv[3]
    output_file = sys.argv[4]

    metadata = os.path.join(data_dir, "metadata.csv")
    #print("Loading Meta-data")
    meta_df = pd.read_csv(metadata, low_memory=False)
    #print("Meta-data loaded")
    meta_df = meta_df[['cord_uid', 'title', 'abstract', 'pdf_json_files', 'pmc_json_files']]
    meta_df['pdf_json_files'] = meta_df['pdf_json_files'].astype(str)
    meta_df['pmc_json_files'] = meta_df['pmc_json_files'].astype(str)

    idf_dict, filemap = get_idf_dict_new()
    #dumpdicts(idf_dict, filemap)

    #print("Loading dictionary files")
    # with open("idf_dict.json", "r") as jsonfile:
    #     idf_dict = json.load(jsonfile)
    # with open("filemap.json", "r") as jsonfile:
    #     filemap = json.load(jsonfile)

    #print("Creating vocab")
    vocab = create_vocab_dict(idf_dict)

    N = len(filemap)

    get_queries()

    top100_dict = get_top100()

    irrelvec = getIrrelVec()

    #print("Updating queries")
    #updated_queries = rocchio_updateQueriesTest(1.0, 0.75, 0.15)
    updated_queries = rocchio_updateQueriesTest(2.0, 0.15, 0.2)

    getUpdatedOrder(updated_queries)
