# imports
import networkx as nx
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
from tqdm import tqdm
from itertools import combinations
from PorterStemmer import *
ps = PorterStemmer()
delim = '''[ ',(){}.@#+!_~&*%^=`|$:;"`\n]'''
delims = [' ', ",", ".", ":", ";", "'", "\"", "@", "#", "+", "!", "_", "~", "&", "*", "%", "^", "=", "`", "|", "$", "\n", "(", ")", ">", "<"]

show_progress = True
consider_double = True
data_dir = r"C:\Files\a3data\20news-bydate-test"
sim_func = 1  # 1: cosine, 2: jaccard
fname_edges = f"similarity_{'cosine' if sim_func==1 else 'jaccard'}.txt"
fname_pagerank = f"pagerank_{'cosine' if sim_func==1 else 'jaccard'}.txt"
bool_dump_page_rank = True
bool_dump_both = False
bool_take_arguments = False


def getTokensFromText(string):
    for delim in delims:
        string = string.replace(delim, " ")

    terms = string.split()
    terms = [ps.stem(term.lower(), 0, len(term)-1) for term in terms]
    return terms


def process_data():
    data = {}
    for topic in tqdm(os.listdir(data_dir), disable=not show_progress):
        topic_path = os.path.join(data_dir, topic)
        for uid in os.listdir(topic_path):
            full_path = os.path.join(topic_path, uid)
            with open(full_path, encoding='latin1') as filedata:
                filedata = filedata.read()
                tokens = getTokensFromText(filedata)
                data[topic+"/"+uid] = tokens
    return data


def get_idf_dict_new():
    idf_dict = {}
    for uid in tqdm(data, disable=not show_progress):
        tokens = set(data[uid])
        for token in tokens:
            if token not in idf_dict:
                idf_dict[token] = 1
            else:
                idf_dict[token] += 1
    return idf_dict


def create_vocab_dict(idf_dict):
    vocab = {}
    for i, word in enumerate(idf_dict.keys()):
        vocab[word] = i
    return vocab


def get_idf(term):
    return math.log(1+N/idf_dict[term], 2)


def get_tf_dict(docid):
    tf_dict = {}
    tokens = data[docid]
    for token in tokens:
        if token not in tf_dict:
            tf_dict[token] = 1
        else:
            tf_dict[token] += 1
    return tf_dict


def getDocVector(docid):
    tf_dict = get_tf_dict(docid)
    vec = np.zeros(len(vocab))
    for word, tf in tf_dict.items():
        if word in vocab:
            pos = vocab[word]
            normtf = math.log(1+tf, 2)
            normidf = get_idf(word)
            tfidf = normtf * normidf
            vec[pos] = tfidf
    return vec


def precomputeDocVecs():
    vecs = {}
    for uid in tqdm(alluids):
        vec = getDocVector(uid)
        vecs[uid] = vec
    return vecs


def createUIDtoNumMapping():
    mapping = {}
    for i, uid in enumerate(alluids):
        mapping[uid] = i
    return mapping


def getDotProdMat():
    M = []
    for uid in (alluids):
        M.append(uid_to_vec[uid])
    M = np.array(M)
    cos_mat = np.dot(M, M.T)
    norm = np.sqrt(1/np.diag(cos_mat))
    mat = cos_mat * norm
    mat = mat.T * norm
    return mat


# def getCosineSim(uid1, uid2):
#     docVec1 = uid_to_vec[uid1]
#     docVec2 = uid_to_vec[uid2]
#     if np.linalg.norm(docVec1) == 0 or np.linalg.norm(docVec2) == 0:
#         return 0.0
#     return np.dot(docVec1, docVec2)/(np.linalg.norm(docVec1) * np.linalg.norm(docVec2))

def getCosineSim(uid1, uid2):
    return dotProdMat[uid_to_num[uid1]][uid_to_num[uid2]]


def getJacobianSim(uid1, uid2):
    tokens1 = set(data[uid1])
    tokens2 = set(data[uid2])
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection)/len(union)


def getWeightedEdges(similarity):
    edges = list(combinations(alluids, 2))
    weighted_edges = []
    for e in tqdm(edges, disable=not show_progress):
        v1 = e[0]
        v2 = e[1]
        if similarity == 1:
            w = getCosineSim(v1, v2)
        else:
            w = getJacobianSim(v1, v2)
        weighted_edges.append((v1, v2, w))
        if consider_double:
            weighted_edges.append((v2, v1, w))
    return weighted_edges


def dump_edges(edges, filename):
    with open(filename, 'w') as f:
        for e in edges:
            if e[2] > 0.0:
                f.write(str(e[0]) + ' ' + str(e[1]) + ' ' + str(round(e[2], 4)) + '\n')


def create_graph(weighted_edges):
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)
    return G


def get_pagerank(G):
    pagerank = nx.pagerank(G, alpha=0.85)
    return sorted(pagerank.items(), key=lambda x: x[1], reverse=True)


def dump_pagerank(pagerank, filename):
    with open(filename, 'w') as f:
        for uid, pr in pagerank[:20]:
            f.write(str(uid) + ' ' + str(pr) + '\n')


if __name__ == "__main__":

    if bool_take_arguments:
        sim = sys.argv[1]
        if sim == 'cosine':
            sim_func = 1
        elif sim == 'jaccard':
            sim_func = 2

        data_dir = sys.argv[2]
        fname_edges = sys.argv[3]
        bool_dump_both = False
        bool_dump_page_rank = False

    data = process_data()
    idf_dict = get_idf_dict_new()
    N = len(data)
    alluids = list(data.keys())

    vocab = create_vocab_dict(idf_dict)
    uid_to_vec = precomputeDocVecs()
    uid_to_num = createUIDtoNumMapping()
    dotProdMat = getDotProdMat()
    weights = getWeightedEdges(sim_func)
    G = create_graph(weights)
    pagerank = get_pagerank(G)
    dump_edges(weights, fname_edges)
    if bool_dump_page_rank:
        dump_pagerank(pagerank, fname_pagerank)

    del pagerank
    del weights

    if bool_dump_both:
        sim_func = 1 if sim_func == 2 else 2
        fname_edges = f"similarity_{'cosine' if sim_func==1 else 'jaccard'}.txt"
        fname_pagerank = f"pagerank_{'cosine' if sim_func==1 else 'jaccard'}.txt"
        weights = getWeightedEdges(sim_func)
        G = create_graph(weights)
        pagerank = get_pagerank(G)
        dump_edges(weights, fname_edges)
        if bool_dump_page_rank:
            dump_pagerank(pagerank, fname_pagerank)
