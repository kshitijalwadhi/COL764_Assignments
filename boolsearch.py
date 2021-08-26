from bs4 import BeautifulSoup
import re
import os
from PorterStemmer import *
from tqdm import tqdm
import snappy
import mmap
import math
import json
import sys
ps = PorterStemmer()


# returns tokens after stemming in the form of a list
def getTokensFromText(text):
    words = re.split('''[ ',.:_;"’`\n]''', text)
    res = []
    for w in words:
        if(len(w) > 0):
            temp = ps.stem(w.lower(), 0, len(w)-1)
            if temp in dictionary:
                res.append(temp)
    return res


# does gap encoding on a list
def gapEncodeList(arr):
    first = arr[0]
    res = [first]
    for i, item in enumerate(arr):
        if i == 0:
            continue
        res.append(item - first)
    return res


# undo gap encoding on a list
def undoGapEncode(arr):
    first = arr[0]
    for i, item in enumerate(arr):
        if i == 0:
            continue
        arr[i] += first
    return arr


# does gap encoding on the inverted index
def doGapEncoding(invidx):
    for token, posting_list in invidx.items():
        invidx[token] = gapEncodeList(posting_list)
    return invidx


def dec_to_binary(n):
    return bin(n).replace("0b", "")


def bin_to_dec(n):
    return int(n, 2)


# performs vb encoding on a number and returns a list of chunks of size 8
def vbencode_number(number):
    bytes_list = []
    while True:
        bytes_list.insert(0, number % 128)
        if number < 128:
            break
        number = number // 128
    for i in range(len(bytes_list)-1):
        bytes_list[i] += 128

    temp = []
    for num in bytes_list:
        temp.append(dec_to_binary(num))
    app = 8 - len(temp[-1])
    temp[-1] = '0'*app + temp[-1]
    return temp


# VB encodes a list of numbers
def vbencode(numbers):
    stream = []
    for n in numbers:
        temp = vbencode_number(n)
        stream.extend(temp)
    return stream


# VB decodes a list of numbers
def vbdecode(stream):
    numbers = []
    n = 0
    for byte in stream:
        if(byte[0] == '1'):
            n = 128*n + bin_to_dec(byte[1:])
        else:
            n = 128*n + bin_to_dec(byte)
            numbers.append(n)
            n = 0
    return numbers


# does c2 encoding on a number
def c2_encoding(x):
    if(x == 1):
        return "0"

    def l(x):
        return len(dec_to_binary(x))

    def U(n):
        if(n <= 0):
            return '0'
        return '1' * (n-1) + '0'

    def lsb(a, b):
        binary = dec_to_binary(a)
        return binary[-b:]
    lx = l(x)
    llx = l(lx)
    t1 = U(llx)
    t2 = lsb(lx, llx-1)
    t3 = lsb(x, lx-1)
    return t1+t2+t3


# does c2 encoding on a list of numbers
def c2_encode_list(arr):
    ans = ''
    for e in arr:
        ans += c2_encoding(e)
    return ans


# does c3 decoding on a list of numbers
def c2_decode(x):
    i = 0
    n = len(x)
    res = []
    while(i < n):
        t1 = ''
        while(x[i] != '0'):
            t1 += x[i]
            i += 1
        llx = len(t1) + 1
        i += 1
        lx_bin = '1' + x[i:i+llx-1]
        i += (llx-1)
        lx = bin_to_dec(lx_bin)
        num_bin = '1' + x[i:i+lx-1]
        i += (lx-1)
        num = bin_to_dec(num_bin)
        res.append(num)
    return res


# chunks a string into desired lengths
def chunkstring(string, length):
    return [string[0+i:length+i] for i in range(0, len(string), length)]


# does c3 encoding on a text
def c3_encoding(text):
    text = str(text)
    return snappy.compress(text)


# c3 encodes a list
def c3_encode_list(arr):
    temp = vbencode(arr)
    s = ''
    for enc in temp:
        s += enc
    return c3_encoding(s)


# c3 decodes a string
def c3_decode_list(text):
    allbytes = snappy.decompress(text).decode('utf-8')
    chunks = chunkstring(allbytes, 8)
    return vbdecode(chunks)


# get posting list using C1
def getPostings_C1(tokens):
    res = []
    for token in tokens:
        start = dictionary[token][0]
        end = start + dictionary[token][1]
        allbytes = mm[start:end]
        allbytes = str(allbytes)
        allbytes = allbytes.replace("'", "")
        allbytes = allbytes.replace("b", "")
        chunks = chunkstring(allbytes, 8)
        doclist = vbdecode(chunks)
        res.append(undoGapEncode(doclist))
    return res


# get posting list using C2
def getPostings_C2(tokens):
    res = []
    for token in tokens:
        start = dictionary[token][0]
        end = start + dictionary[token][1]
        allbytes = mm[start:end]
        allbytes = str(allbytes)
        allbytes = allbytes.replace("'", "")
        allbytes = allbytes.replace("b", "")
        doclist = c2_decode(allbytes)
        res.append(undoGapEncode(doclist))
    return res


# get posting list using C3
def getPostings_C3(tokens):
    res = []
    for token in tokens:
        start = dictionary[token][0]
        end = start + dictionary[token][1]
        allbytes = mm[start:end]
        doclist = c3_decode_list(allbytes)
        res.append(undoGapEncode(doclist))
    return res


# find list of common docs from multiple lists
def getIntersection(postings):
    ans = postings[0]
    for posting in postings:
        tempans = []
        n1 = len(ans)
        n2 = len(posting)
        i, j = 0, 0
        while i != n1 and j != n2:
            if ans[i] == posting[j]:
                tempans.append(ans[i])
                i += 1
                j += 1
            elif ans[i] < posting[j]:
                i += 1
            else:
                j += 1
        ans = tempans
    return ans


if __name__ == "__main__":
    queryfile = sys.argv[1]
    resultfile = sys.argv[2]
    indexfile = sys.argv[3]
    dictfile = sys.argv[4]

    with open(dictfile, "r") as jsonfile:
        tempdict = json.load(jsonfile)

    compression = tempdict["compression"]
    dictionary = tempdict["dictionary"]
    int_to_docID = tempdict["int_to_docID"]

    f = open(indexfile, "r+b")
    mm = mmap.mmap(f.fileno(), 0)

    queries = open(queryfile, "r")
    lines = queries.readlines()

    for line in lines:
        if compression == 1:
            postings = getPostings_C1(getTokensFromText(line))
        elif compression == 2:
            postings = getPostings_C2(getTokensFromText(line))
        elif compression == 3:
            postings = getPostings_C3(getTokensFromText(line))

        docids = getIntersection(postings)
        ans = []
        for doc in docids:
            ans.append(int_to_docID[str(doc)])
        print(ans)

    queries.close()
    f.close()
