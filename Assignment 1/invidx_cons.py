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
import time
ps = PorterStemmer()

delim = '''[ ',(){}.:;"’`\n]'''


# returns tokens after stemming in the form of a list
def getTokensFromText(text):
    words = re.split(delim, text)
    res = []
    for w in words:
        if(len(w) > 0):
            res.append(ps.stem(w.lower(), 0, len(w)-1))
    return res


# returns a dictionary containing stopwords
def processStopwords(filepath):
    stopwords = {}
    text = ''
    file = open(filepath, "r")
    lines = file.readlines()
    for line in lines:
        text += (line + " ")
    res = re.split(delim, text)
    temp = []
    for w in res:
        if len(w) > 0:
            temp.append(w.lower())
    res = temp
    for w in res:
        stopwords[w] = 1
    return stopwords


# check if word present in stopwords
def checkStopwords(word, stopwords):
    if stopwords.get(word) is not None:
        return True
    else:
        return False


# get list of XML tags present in the xml-tags-info file
def getXMLtags(filepath):
    file = open(filepath, "r")
    lines = file.readlines()
    text = ''
    for line in lines:
        text += (line + " ")
    res = re.split(delim, text)
    temp = []
    for w in res:
        if len(w) > 0:
            temp.append(w.lower())
    res = temp
    return res


# maps docnum to list of tokens for all files in directory
def process_directory(dir_path):
    data = {}
    for filename in tqdm(os.listdir(dir_path)):
        # if filename == "ap890520":
        #     continue
        full_path = os.path.join(dir_path, filename)
        file = open(full_path, "r")
        contents = file.read()
        soup = BeautifulSoup(contents, 'html.parser')
        docs = soup.find_all('doc')

        for doc in docs:
            docnum = doc.find(xmlTags[0]).get_text().strip()
            res = []
            for tag in xmlTags:
                if tag == xmlTags[0]:
                    continue
                fields = doc.find_all(tag)
                for field in fields:
                    text = field.get_text()
                    words = re.split(delim, text)
                    for w in words:
                        if(len(w) > 0):
                            res.append(ps.stem(w.lower(), 0, len(w)-1))
            data[docnum] = res
    return data


# gets vocabulary from the data and returns it in the form of a set
def getVocab(data, stopwords):
    tokens = []
    for doc, token_list in data.items():
        tokens += token_list
    tokens = set(tokens)
    for w in stopwords:
        tokens.discard(w)
    return tokens


# takes a list of document names and maps them to integers and returns the map
def mapDocIDs(docIDs):
    docID_to_int = {}
    int_to_docID = {}
    i = 1
    for doc in docIDs:
        docID_to_int[doc] = i
        int_to_docID[i] = doc
        i += 1
    return docID_to_int, int_to_docID


def makeDocIdMap(full_path):
    file = open(full_path, "r")
    contents = file.read()
    soup = BeautifulSoup(contents, 'html.parser')
    docs = soup.find_all('doc')
    docIDs = []
    for doc in docs:
        docnum = doc.find('docno').get_text().strip()
        docIDs.append(docnum)
    return mapDocIDs(docIDs)


# generates inverted index from the given data
def getInvIdx(data):
    invidx = {}
    for doc, token_list in data.items():
        doc = docID_to_int[doc]
        for token in token_list:
            if token in vocab:
                if token in invidx.keys():
                    if(invidx[token][-1] != doc):
                        invidx[token].append(doc)
                else:
                    invidx[token] = [doc]
    return invidx


# generates inverted index from the given data
def getInvIdx(data):
    invidx = {}
    for doc, token_list in data.items():
        doc = docID_to_int[doc]
        for token in token_list:
            if token in vocab:
                if token in invidx.keys():
                    if(invidx[token][-1] != doc):
                        invidx[token].append(doc)
                else:
                    invidx[token] = [doc]
    return invidx


# does gap encoding on a list
def gapEncodeList(arr):
    carry = arr[0]
    temp = [carry]
    for i in range(1, len(arr)):
        temp.append(arr[i]-arr[i-1])
    return temp


# undo gap encoding on a list
def undoGapEncode(arr):
    temp = [arr[0]]
    for i in range(1, len(arr)):
        prev = temp[-1]
        temp.append(arr[i]+prev)
    return temp


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


# makes posting list file for C1 compression
def dumpFiles_C1(invidx):
    dictionary = {}
    fname = indexfile + '.idx'
    file = open(fname, "wb")
    offset = 0
    for term, posting_list in invidx.items():
        encoded = vbencode(posting_list)
        temp = []
        for enc in encoded:
            temp.append(bin_to_dec(enc))
        file.write(bytearray(temp))
        length = len(temp)
        dictionary[term] = [offset, length]
        offset += length
    file.close()
    return dictionary


# does c2 encoding on a number
def c2_encoding(x):
    if(x == 1):
        return "0"

    def l(x):
        return math.floor(math.log2(x))+1

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


# makes posting list file for C2 compression
def dumpFiles_C2(invidx):
    dictionary = {}
    fname = indexfile + '.idx'
    file = open(fname, "wb")
    offset = 0
    for term, posting_list in invidx.items():
        allbytes = c2_encode_list(posting_list)
        sz = len(allbytes)
        if(sz % 8 != 0):
            allbytes = '0'*(8-sz % 8) + allbytes
        skipbits = (8-sz % 8)
        chunks = chunkstring(allbytes, 8)
        temp = []
        for chunk in chunks:
            temp.append(bin_to_dec(chunk))
        file.write(bytearray(temp))
        length = len(chunks)
        dictionary[term] = [offset, length, skipbits]
        offset += length
    file.close()
    return dictionary


# chunks a string into desired lengths
def chunkstring(string, length):
    return [string[0+i:length+i] for i in range(0, len(string), length)]


# makes posting list file for C3 compression
def dumpFiles_C3(invidx):
    dictionary = {}
    fname = indexfile + '.idx'
    file = open(fname, "wb")
    offset = 0
    cont = []
    for term, posting_list in tqdm(invidx.items()):
        temp = c2_encode_list(posting_list)
        invidx[term] = []
        cont.append(temp)
        length = len(temp)
        dictionary[term] = [offset, length]
        offset += length
    allbytes = "".join(cont)
    sz = len(allbytes)
    if(sz % 8 != 0):
        allbytes = '0'*(8-sz % 8) + allbytes
    chunks = chunkstring(allbytes, 8)
    temp = []
    for chunk in chunks:
        temp.append(bin_to_dec(chunk))
    file.write(snappy.compress(bytearray(temp)))
    file.close()
    return dictionary, (8-sz % 8)


# does C4 encoding on x using k as param, returns string
def c4_encoding(x, k):
    def U(n):
        if(n <= 0):
            return ''
        return '1' * (n-1) + '0'
    b = pow(2, k)
    q = math.floor((x-1)/b)
    r = x - q*b - 1
    t1 = U(q+1)
    temp = dec_to_binary(r)
    t2 = '0'*(k-len(temp)) + temp
    return t1 + t2


# does C4 encoding on a list
def c4_encode_list(arr, k):
    ans = ''
    for e in arr:
        ans += c4_encoding(e, k)
    return ans


# does C4 decoding with numbits given
def c4_decode_withpad(stream, k, numbits):
    i = 0
    b = pow(2, k)
    res = []
    while(i < numbits):
        q = 0
        while(stream[i] != '0'):
            q += 1
            i += 1
        i += 1
        r_bin = stream[i:i+k]
        i += k
        r = bin_to_dec(r_bin)
        x = q*b + r + 1
        # print(x)
        res.append(x)
    return res


# Makes posting list file for C4 compression
def dumpFiles_C4(invidx):
    dictionary = {}
    fname = indexfile + '.idx'
    file = open(fname, "wb")
    offset = 0
    for term, posting_list in invidx.items():
        maxm = posting_list[-1]
        k = int(math.log2(maxm))
        allbytes = c4_encode_list(posting_list, k)
        numbits = len(allbytes)
        if(numbits % 8 != 0):
            allbytes = allbytes + '0'*(8-numbits % 8)
        chunks = chunkstring(allbytes, 8)
        temp = []
        for chunk in chunks:
            temp.append(bin_to_dec(chunk))
        file.write(bytearray(temp))
        dictionary[term] = [offset, numbits, k]
        offset += len(temp)
    file.close()
    return dictionary


# dumps the dictionary containing offsets and also the int_to_docID map into a json
def dumpDicts(dictionary, int_to_docID):
    fname = indexfile + '.dict'
    with open(fname, "w") as outfile:
        res = {"compression": compression, "orig_offset": orig_offset, "dictionary": dictionary, "int_to_docID": int_to_docID}
        json.dump(res, outfile)


if __name__ == "__main__":
    coll_path = sys.argv[1]
    indexfile = sys.argv[2]
    stopwordfile = sys.argv[3]
    compression = int(sys.argv[4])
    xmltagsinfo = sys.argv[5]

    start = time.time()

    stopwords = processStopwords(stopwordfile)
    xmlTags = getXMLtags(xmltagsinfo)
    data = process_directory(coll_path)

    print(f"Processed data. {time.time() - start} since start")

    docID_to_int, int_to_docID = mapDocIDs(data)
    vocab = getVocab(data, stopwords)

    invidx = getInvIdx(data)
    del data
    print(f"Created Inverted Index. {time.time() - start} since start")

    invidx = doGapEncoding(invidx)

    orig_offset = 0

    if compression == 0 or compression == 1:
        dictionary = dumpFiles_C1(invidx)
    elif compression == 2:
        dictionary = dumpFiles_C2(invidx)
    elif compression == 3:
        dictionary, orig_offset = dumpFiles_C3(invidx)
    elif compression == 4:
        dictionary = dumpFiles_C4(invidx)

    dumpDicts(dictionary, int_to_docID)

    end = time.time()
    print(end-start)
