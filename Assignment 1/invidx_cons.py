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
FILES_PER_SPLIT = 100


# returns tokens after stemming in the form of a list (CHECKED)
def getTokensFromText(text):
    words = re.split(delim, text)
    res = []
    for w in words:
        if(len(w) > 0):
            res.append(ps.stem(w.lower(), 0, len(w)-1))
    return res


# returns a dictionary containing stopwords (CHECKED)
def processStopwords(filepath):
    stopwords = {}
    text = ''
    file = open(filepath, "r")
    lines = file.readlines()
    # for line in lines:
    #     text += (line + " ")
    text = " ".join(lines)
    res = re.split(delim, text)
    temp = []
    for w in res:
        if len(w) > 0:
            temp.append(ps.stem(w.lower(), 0, len(w)-1))
    res = set(temp)
    return res


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
    # for line in lines:
    #     text += (line + " ")
    text = " ".join(lines)
    res = re.split(delim, text)
    temp = []
    for w in res:
        if len(w) > 0:
            temp.append(w.lower())
    res = temp
    return res


def process_directory(dir_path, filearr):
    data = {}
    for filename in tqdm(filearr):
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


# maps docnum to list of tokens for all files in directory
def process_directory_old(dir_path):
    data = {}
    for filename in tqdm(os.listdir(dir_path)):
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
def getVocab(vocab, data, stopwords):
    tokens = []
    for doc, token_list in data.items():
        tokens += token_list
    tokens = set(tokens)
    for w in stopwords:
        tokens.discard(w)
    vocab.update(tokens)
    return vocab


def getVocab_old(data, stopwords):
    tokens = []
    for doc, token_list in data.items():
        tokens += token_list
    tokens = set(tokens)
    for w in stopwords:
        tokens.discard(w)
    return tokens


# takes a list of document names and maps them to integers and returns the map
def mapDocIDs(docIDs, numdocs):
    for doc in docIDs:
        docID_to_int[doc] = numdocs
        int_to_docID[numdocs] = doc
        numdocs += 1
    return numdocs


def mapDocIDs_old(docIDs):
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


def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


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


def c0_encode(x):
    binary = dec_to_binary(x)
    binary = binary.zfill(32)
    return binary


def c0_encode_list(arr):
    temp = []
    for e in arr:
        temp.append(c0_encode(e))
    return "".join(temp)


def c0_decode(data):
    allbin = chunkstring(data, 32)
    temp = []
    for e in allbin:
        temp.append(bin_to_dec(e))
    return temp


def dumpFiles_C0():
    dictionary = {}
    alldicts = loadDictFiles()
    allidx = loadBinFiles()
    fname = indexfile + '.idx'
    file = open(fname, "wb")
    offset = 0
    for w in vocab:
        postings = []
        for i, tempdict in enumerate(alldicts):
            if w in tempdict:
                reader = allidx[i]
                postings.extend(getPosting(reader, w, alldicts[i]))
        postings = gapEncodeList(postings)
        allbytes = c0_encode_list(postings)
        sz = len(allbytes)
        chunks = chunkstring(allbytes, 8)
        temp = []
        for chunk in chunks:
            temp.append(bin_to_dec(chunk))
        file.write(bytearray(temp))
        length = len(chunks)
        dictionary[w] = [offset, length]
        offset += length
    file.close()
    closeBinFiles(allidx)
    return dictionary


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
    #app = 8 - len(temp[-1])
    #temp[-1] = '0'*app + temp[-1]
    temp[-1] = temp[-1].zfill(8)
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
def dumpFiles_C1():
    dictionary = {}
    alldicts = loadDictFiles()
    allidx = loadBinFiles()
    fname = indexfile + '.idx'
    file = open(fname, "wb")
    offset = 0
    for w in vocab:
        postings = []
        for i, tempdict in enumerate(alldicts):
            if w in tempdict:
                reader = allidx[i]
                postings.extend(getPosting(reader, w, alldicts[i]))
        postings = gapEncodeList(postings)
        encoded = vbencode(postings)
        temp = []
        for enc in encoded:
            temp.append(bin_to_dec(enc))
        file.write(bytearray(temp))
        length = len(temp)
        dictionary[w] = [offset, length]
        offset += length
    file.close()
    closeBinFiles(allidx)
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
    temp = []
    for e in arr:
        #ans += c2_encoding(e)
        temp.append(c2_encoding(e))
    ans = "".join(temp)
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
def dumpFiles_C2():
    dictionary = {}
    alldicts = loadDictFiles()
    allidx = loadBinFiles()
    fname = indexfile + '.idx'
    file = open(fname, "wb")
    offset = 0
    for w in vocab:
        postings = []
        for i, tempdict in enumerate(alldicts):
            if w in tempdict:
                reader = allidx[i]
                postings.extend(getPosting(reader, w, alldicts[i]))
        postings = gapEncodeList(postings)
        allbytes = c2_encode_list(postings)
        sz = len(allbytes)
        skipbits = 0
        if(sz % 8 != 0):
            allbytes = '0'*(8-sz % 8) + allbytes
            skipbits = (8-sz % 8)
        chunks = chunkstring(allbytes, 8)
        temp = []
        for chunk in chunks:
            temp.append(bin_to_dec(chunk))
        file.write(bytearray(temp))
        length = len(chunks)
        dictionary[w] = [offset, length, skipbits]
        offset += length
    closeBinFiles(allidx)
    file.close()
    return dictionary


# chunks a string into desired lengths
def chunkstring(string, length):
    return [string[0+i:length+i] for i in range(0, len(string), length)]


def c3encodehelper(arr):
    temp = []
    for e in arr:
        temp.append(str(e))
    return ",".join(temp)


def c3decodehelper(string):
    ans = string.split(",")
    return list(map(int, ans))


# makes posting list file for C3 compression
def dumpFiles_C3(invidx):
    dictionary = {}
    fname = indexfile + '.idx'
    file = open(fname, "wb")
    offset = 0
    cont = []
    for term, posting_list in invidx.items():
        temp = c3encodehelper(posting_list)
        cont.append(temp)
        invidx[term] = []
        length = len(temp)
        dictionary[term] = [offset, length]
        offset += length
    allbytes = "".join(cont)
    file.write(snappy.compress(allbytes))
    file.close()
    return dictionary


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
    temp = []
    for e in arr:
        #ans += c4_encoding(e, k)
        temp.append(c4_encoding(e, k))
    ans = "".join(temp)
    return ans


# does C4 decoding with numbits given
def c4_decode_withpad(stream, k, numbits):
    i = 0
    b = pow(2, k)
    res = []
    if k == 0:
        num_bits_in_one = 2
    else:
        num_bits_in_one = k+1
    while(i < numbits):
        while(True):
            if(i+num_bits_in_one-1 < numbits):
                if(stream[i:i+num_bits_in_one] == ("0"*num_bits_in_one)):
                    res.append(1)
                    i += num_bits_in_one
                else:
                    break
            else:
                break
        if(i >= numbits):
            break
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
def dumpFiles_C4():
    dictionary = {}
    alldicts = loadDictFiles()
    allidx = loadBinFiles()
    fname = indexfile + '.idx'
    file = open(fname, "wb")
    offset = 0
    for w in vocab:
        postings = []
        for i, tempdict in enumerate(alldicts):
            if w in tempdict:
                reader = allidx[i]
                postings.extend(getPosting(reader, w, alldicts[i]))
        postings = gapEncodeList(postings)
        maxm = max(postings)
        k = int(math.log2(maxm))
        allbytes = c4_encode_list(postings, k)
        numbits = len(allbytes)
        if(numbits % 8 != 0):
            allbytes = allbytes + '0'*(8-numbits % 8)
        chunks = chunkstring(allbytes, 8)
        temp = []
        for chunk in chunks:
            temp.append(bin_to_dec(chunk))
        file.write(bytearray(temp))
        dictionary[w] = [offset, numbits, k, len(temp)]
        offset += len(temp)
    closeBinFiles(allidx)
    file.close()
    return dictionary


# dumps the dictionary containing offsets and also the int_to_docID map into a json
def dumpDicts(dictionary, int_to_docID):
    fname = indexfile + '.dict'
    with open(fname, "w") as outfile:
        res = {"compression": compression, "dictionary": dictionary, "int_to_docID": int_to_docID}
        json.dump(res, outfile)


def padding(x):
    #n = len(x)
    # return '0'*(8-n) + x
    return x.zfill(8)


# makes posting list file using C2 compression
def dumpINV(invidx, i):
    dictionary = {}
    fname = os.path.join("tempfiles", f"index{i}.idx")
    file = open(fname, "wb")
    offset = 0
    for term, posting_list in invidx.items():
        allbytes = c2_encode_list(posting_list)
        sz = len(allbytes)
        skipbits = 0
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


# get posting list using C2
def getPosting(reader, token, tempdict):
    start = tempdict[token][0]
    sz = tempdict[token][1]
    skipbits = tempdict[token][2]
    reader.seek(start)
    allbytes = ''
    vararr = []
    for i in range(0, sz):
        temp = reader.read(1)
        val = int.from_bytes(temp, "big")
        #allbytes += padding(dec_to_binary(val))
        vararr.append(padding(dec_to_binary(val)))
    allbytes = "".join(vararr)
    i = skipbits
    allbytes = allbytes[i:]
    doclist = c2_decode(allbytes)
    return undoGapEncode(doclist)


def loadDictFiles():
    res = []
    for i in range(numfiles):
        with open(os.path.join("tempfiles", f"dict{i}.json"), "r") as jsonfile:
            tempinvidx = json.load(jsonfile)
            res.append(tempinvidx)
    return res


def loadBinFiles():
    res = []
    for i in range(numfiles):
        f = open(os.path.join("tempfiles", f"index{i}.idx"), "rb")
        res.append(f)
    return res


def closeBinFiles(allidx):
    for i in range(numfiles):
        allidx[i].close()


if __name__ == "__main__":
    coll_path = sys.argv[1]
    indexfile = sys.argv[2]
    stopwordfile = sys.argv[3]
    compression = int(sys.argv[4])
    xmltagsinfo = sys.argv[5]

    start = time.time()

    stopwords = processStopwords(stopwordfile)
    xmlTags = getXMLtags(xmltagsinfo)

    if compression == 3:
        data = process_directory_old(coll_path)
        print(f"Processed data. {time.time() - start} since start")

        docID_to_int, int_to_docID = mapDocIDs_old(data)
        vocab = getVocab_old(data, stopwords)

        invidx = getInvIdx(data)
        del data
        print(f"Created Inverted Index. {time.time() - start} since start")
        invidx = doGapEncoding(invidx)

        dictionary = dumpFiles_C3(invidx)
    else:
        allfiles = os.listdir(coll_path)
        int_to_docID = {}
        numfiles = len(chunks(allfiles, FILES_PER_SPLIT))
        numdocs = 1
        docID_to_int = {}
        int_to_docID = {}
        vocab = set()

        for i, chunked in enumerate(chunks(allfiles, FILES_PER_SPLIT)):
            data = process_directory(coll_path, chunked)
            numdocs = mapDocIDs(data, numdocs)
            vocab = getVocab(vocab, data, stopwords)
            invidx = getInvIdx(data)
            invidx = doGapEncoding(invidx)
            tempdict = dumpINV(invidx, i)
            with open(os.path.join("tempfiles", f"dict{i}.json"), "w") as outfile:
                json.dump(tempdict, outfile)
            invidx.clear()
            tempdict.clear()
            del tempdict
            del invidx

        print(f"Processed data. {time.time() - start} since start")

        if compression == 0:
            dictionary = dumpFiles_C0()
        elif compression == 1:
            dictionary = dumpFiles_C1()
        elif compression == 2:
            dictionary = dumpFiles_C2()
        elif compression == 4:
            dictionary = dumpFiles_C4()

    dumpDicts(dictionary, int_to_docID)

    end = time.time()
    print(end-start)
