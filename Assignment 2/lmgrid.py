# coding: utf-8
from nltk.corpus import stopwords
import random
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
ps = PorterStemmer()
random.seed(42)

stop_words = set(stopwords.words('english'))

delim = '''[ ',(){}.:;"`\n]'''


topics_file = "covid19-topics.xml"
top100_file = "t40-top-100.txt"
relevance_file = "t40-qrels.txt"


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


def get_queries():
    with open(os.path.join(data_dir, topics_file), "r") as file:
        content = file.readlines()
        content = "".join(content)
        bs_content = BeautifulSoup(content, "lxml")
        queries = bs_content.find_all("query")
        all_text = []
        for q in queries:
            all_text.append(q.text)
        return all_text


def get_relevances():
    rel_dict = {}
    with open(os.path.join(data_dir, relevance_file), "r") as file:
        for x in file:
            line = x.split(" ")
            qid = int(line[0])
            docid = line[2]
            rel = int(line[3].replace("\n", ""))
            if qid not in rel_dict:
                rel_dict[qid] = {docid: rel}
            else:
                rel_dict[qid][docid] = rel
    return rel_dict


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


def get_tf_dict(docid):
    tf_dict = {}
    paths = filemap[docid]
    alltext = ""
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
    for token in tokens:
        if token not in tf_dict:
            tf_dict[token] = 1
        else:
            tf_dict[token] += 1
    return tf_dict


def get_filepath():
    filemap = {}
    meta_df2 = pd.read_csv(metadata, low_memory=False)
    meta_df2 = meta_df2[['cord_uid', 'title', 'abstract', 'pdf_json_files', 'pmc_json_files']]
    meta_df2['pdf_json_files'] = meta_df2['pdf_json_files'].astype(str)
    meta_df2['pmc_json_files'] = meta_df2['pmc_json_files'].astype(str)
    for row in tqdm(meta_df2.itertuples(), total=meta_df2.shape[0]):
        uid = row.cord_uid
        alltext = ""
        if row.pmc_json_files:
            paths = row.pmc_json_files
        elif row.pdf_json_files:
            paths = row.pdf_json_files
        filemap[uid] = paths
    return filemap


def getFreqMap(qno):
    # tf map from word->doc-> freq
    top100docs = top100_dict[qno]
    freq_map = {}
    doclen_map = {}
    for doc in (top100docs):
        tf_dict = get_tf_dict(doc)
        total = 0
        for entry in tf_dict.items():
            token = entry[0]
            freq = entry[1]
            total += freq
            if token not in freq_map:
                freq_map[token] = {}
            freq_map[token][doc] = freq
        doclen_map[doc] = total
    vocab = list(freq_map.keys())

    return freq_map, doclen_map


def getProb_dict(qno, mu):
    freq_map, doclen_map = getFreqMap(qno)
    vocab = list(freq_map.keys())
    # map word->doc->prob
    prob_map = {}
    tf_dict = {}
    for w in vocab:
        tot = 0
        for doc in freq_map[w].keys():
            t1 = freq_map[w][doc]
            tot += t1
        tf_dict[w] = tot
    for w in vocab:
        for doc in freq_map[w].keys():
            t1 = freq_map[w][doc]
            t2 = tf_dict[w]
            if doclen_map[doc] + mu == 0:
                prob = 0
            else:
                prob = (t1+mu*t2)/(doclen_map[doc] + mu)
            if w not in prob_map:
                prob_map[w] = {}
            prob_map[w][doc] = prob
    return prob_map


def getTopkExpR1(qno, k, mu):
    prob_map = getProb_dict(qno, mu)
    q = q_map[qno]
    top100docs = top100_dict[qno]
    qtokens = getTokensFromText(q)
    # map from word -> finProb
    finmap = {}
    vocab = list(prob_map.keys())
    for w in vocab:
        total = 0
        for doc in top100docs:
            prob = 1
            for qi in qtokens:
                if doc in prob_map[qi]:
                    prob *= prob_map[qi][doc]
                else:
                    prob = 0
            if doc in prob_map[w]:
                prob *= prob_map[w][doc]
            else:
                prob = 0
            total += prob
        finmap[w] = total
    sorted_tokens = dict(sorted(finmap.items(), key=lambda x: x[1], reverse=True))
    sorted_tokens = list(sorted_tokens.keys())
    return sorted_tokens[:k]


def getTopkExpR2(qno, k, mu):
    prob_map = getProb_dict(qno, mu)
    q = q_map[qno]
    top100docs = top100_dict[qno]
    qtokens = getTokensFromText(q)
    # map from word -> finProb
    finmap = {}
    vocab = list(prob_map.keys())
    for w in vocab:
        total = 0
        for doc in top100docs:
            if doc in prob_map[w]:
                total += prob_map[w][doc]
        p_w = total
        prob = p_w * p_w

        for qi in qtokens:
            total = 0
            for doc in top100docs:
                temp = 1
                if doc in prob_map[w]:
                    temp *= prob_map[w][doc]
                else:
                    temp *= 0
                if doc in prob_map[qi]:
                    temp *= prob_map[qi][doc]
                else:
                    temp *= 0
                total += temp
            prob *= total
        finmap[w] = total
    sorted_tokens = dict(sorted(finmap.items(), key=lambda x: x[1], reverse=True))
    sorted_tokens = list(sorted_tokens.keys())
    return sorted_tokens[:k]


def getQueryVector(mode, query, qno, k, mu, vocab):
    tokens = getTokensFromText(query)
    if mode == 1:
        tokens.extend(getTopkExpR1(qno, k, mu))
    else:
        tokens.extend(getTopkExpR2(qno, k, mu))
    query_tf_dict = {}
    for token in tokens:
        if token not in query_tf_dict:
            query_tf_dict[token] = 1
        else:
            query_tf_dict[token] += 1
    vec = np.zeros(len(vocab))
    freq_map, doclen_map = getFreqMap(qno)
    for word, tf in query_tf_dict.items():
        if word in vocab.keys():
            pos = vocab[word]
            normtf = 1 + math.log(tf, 2)
            #normidf = get_idf(word)
            idf = len(freq_map[word])
            normidf = math.log(1+100/idf, 2)
            tfidf = normtf * normidf
            vec[pos] = tfidf
    return vec


def LM_updateQueries(mode, k, mu):
    updated_queries = []
    for i, q_orig in tqdm(enumerate(get_queries())):
        q = q_orig.lower()
        qno = i+1
        vocab = vocab_dict[qno]
        qv0 = getQueryVector(mode, q, qno, k, mu, vocab)
        updated_queries.append(qv0)
    return updated_queries


def getDocVector(docid, vocab, freq_map):
    tf_dict = get_tf_dict(docid)
    vec = np.zeros(len(vocab))
    for word, tf in tf_dict.items():
        if word in vocab:
            pos = vocab[word]
            normtf = 1 + math.log(tf, 2)
            idf = len(freq_map[word])
            normidf = math.log(1+100/idf, 2)
            tfidf = normtf * normidf
            vec[pos] = tfidf
    return vec


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


def getSim(queryVec, docVec):
    if np.linalg.norm(queryVec) == 0 or np.linalg.norm(docVec) == 0:
        return 0.0
    return np.dot(queryVec, docVec)/(np.linalg.norm(queryVec) * np.linalg.norm(docVec))


def LM_getNDCG(updated_queries):
    total = 0
    allscores = []
    for i, q in tqdm(enumerate(updated_queries)):
        qno = i+1
        freq_map, doclen_map = getFreqMap(qno)
        doc_score_dict = {}
        top100docs = top100_dict[qno]
        vocab = vocab_dict[qno]
        for doc in top100docs:
            score = getSim(q, getDocVector(doc, vocab, freq_map))
            doc_score_dict[doc] = score
        sorted_docs = dict(sorted(doc_score_dict.items(), key=lambda x: x[1], reverse=True))
        sorted_docs = list(sorted_docs.keys())
        ndcg = getNDCG(qno, sorted_docs)
        allscores.append(ndcg)
        total += ndcg
    return allscores, total/len(updated_queries)


def gridSearch():
    res = {'mu': [], 'qid': [], 'nDCG_R1': [], 'nDCG_R2': []}
    avgres = {'mu': [], 'nDCG_R1': [], 'nDCG_R2': []}
    df = pd.DataFrame(res)
    avgdf = pd.DataFrame(avgres)
    gridvals = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for mu in tqdm(gridvals):
        updated_queries_1 = LM_updateQueries(1, topK, mu)
        allscores_1, ndcg_1 = LM_getNDCG(updated_queries_1)
        updated_queries_2 = LM_updateQueries(2, topK, mu)
        allscores_2, ndcg_2 = LM_getNDCG(updated_queries_2)
        mu_arr = [mu] * len(updated_queries_1)
        q_arr = np.arange(1, len(updated_queries_1)+1, 1)
        df2 = pd.DataFrame({'mu': mu_arr, 'qid': q_arr, 'nDCG_R1': allscores_1, 'nDCG_R2': allscores_2})
        df = df.append(df2, ignore_index=True)
        avgdf = avgdf.append(pd.DataFrame({'mu': [mu], 'nDCG_R1': [ndcg_1], 'nDCG_R2': [ndcg_2]}))
    df.to_csv('query_wise_LM.csv', index=False)
    avgdf.to_csv('avg_LM.csv', index=False)


if __name__ == "__main__":

    data_dir = "C:\Files\col764-a2-release\col764-ass2-release\data"
    #data_dir = "/home/ubuntu/satpop/kshitijExp/Data"

    metadata = os.path.join(data_dir, "metadata.csv")
    print("Loading Meta-data")
    meta_df = pd.read_csv(metadata, low_memory=False)
    print("Meta-data loaded")
    meta_df = meta_df[['cord_uid', 'title', 'abstract', 'pdf_json_files', 'pmc_json_files']]
    meta_df['pdf_json_files'] = meta_df['pdf_json_files'].astype(str)
    meta_df['pmc_json_files'] = meta_df['pmc_json_files'].astype(str)

    print("Creating Filemap")
    filemap = get_filepath()
    print("Creating top100 dict")
    top100_dict = get_top100()
    print("Creating relevance dict")
    rel_dict = get_relevances()

    print("Processing queries file")
    q_map = {}
    for i, q_orig in (enumerate(get_queries())):
        q = q_orig.lower()
        q_map[i+1] = q

    print("Processing queries")
    vocab_dict = {}
    for i, q_orig in tqdm(enumerate(get_queries())):
        qno = i+1
        freq_map, doclen_map = getFreqMap(qno)
        vocab = {}
        for i, word in enumerate(freq_map.keys()):
            vocab[word] = i
        vocab_dict[qno] = vocab

    #mode = 1
    topK = 20
    mu = 0.6

    gridSearch()
