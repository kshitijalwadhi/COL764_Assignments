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

delim = '''[ ',(){}.:;"’`\n]'''


# returns tokens after stemming in the form of a list
def getTokensFromText(text):
    words = re.split(delim, text)
    res = []
    for w in words:
        if(len(w) > 0):
            temp = ps.stem(w.lower(), 0, len(w)-1)
            res.append(temp)
    return res


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


# does c2 decoding on a list of numbers
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


def c3encodehelper(arr):
    temp = []
    for e in arr:
        temp.append(str(e))
    return ",".join(temp)


def c3decodehelper(string):
    ans = string.split(",")
    return list(map(int, ans))


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


# chunks a string into desired lengths
def chunkstring(string, length):
    return [string[0+i:length+i] for i in range(0, len(string), length)]


def padding(x):
    n = len(x)
    return '0'*(8-n) + x


# get posting list using C0
def getPostings_C0(tokens,):
    for token in tokens:
        if token not in dictionary:
            return []
    res = []
    for token in tokens:
        start = dictionary[token][0]
        sz = dictionary[token][1]
        f.seek(start)
        allbytes = ''
        for i in range(0, sz):
            temp = f.read(1)
            val = int.from_bytes(temp, "big")
            allbytes += padding(dec_to_binary(val))
        doclist = c0_decode(allbytes)
        res.append(undoGapEncode(doclist))
    return res


# get posting list using C1
def getPostings_C1(tokens):
    for token in tokens:
        if token not in dictionary:
            return []
    res = []
    for token in tokens:
        start = dictionary[token][0]
        sz = dictionary[token][1]
        f.seek(start)
        allbytes = ''
        for i in range(0, sz):
            temp = f.read(1)
            val = int.from_bytes(temp, "big")
            allbytes += padding(dec_to_binary(val))
        chunks = chunkstring(allbytes, 8)
        doclist = vbdecode(chunks)
        res.append(undoGapEncode(doclist))
    return res


# get posting list using C2
def getPostings_C2(tokens):
    for token in tokens:
        if token not in dictionary:
            return []
    res = []
    for token in tokens:
        start = dictionary[token][0]
        sz = dictionary[token][1]
        skipbits = dictionary[token][2]
        f.seek(start)
        allbytes = ''
        for i in range(0, sz):
            temp = f.read(1)
            val = int.from_bytes(temp, "big")
            allbytes += padding(dec_to_binary(val))
        i = skipbits
        allbytes = allbytes[i:]
        doclist = c2_decode(allbytes)
        res.append(undoGapEncode(doclist))
    return res


# get posting list using C3
def getPostings_C3(tokens):
    for token in tokens:
        if token not in dictionary:
            return []
    res = []
    for token in tokens:
        start = dictionary[token][0]
        end = start + dictionary[token][1]
        subset = decoded[start:end]
        subset = str(subset)
        subset = subset.replace("b'", "")
        subset = subset.replace("'", "")
        docList = c3decodehelper(subset)
        res.append(undoGapEncode(docList))
    return res


# get posting list using C4
def getPostings_C4(tokens):
    for token in tokens:
        if token not in dictionary:
            return []
    res = []
    for token in tokens:
        start = dictionary[token][0]
        numbits = dictionary[token][1]
        k = dictionary[token][2]
        f.seek(start)
        allbytes = ''
        numiter = numbits
        numiter += (numiter % 8)
        numiter = int(numiter/8)
        for i in range(0, numiter):
            temp = f.read(1)
            val = int.from_bytes(temp, "big")
            allbytes += padding(dec_to_binary(val))
        doclist = c4_decode_withpad(allbytes, k, numbits)
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

    f = open(indexfile, "rb")

    queries = open(queryfile, "r")
    lines = queries.readlines()

    res_file = open(resultfile, "w")

    if compression == 3:
        encoded = f.read()
        decoded = snappy.decompress(encoded)

    for qid, line in enumerate(lines):
        if compression == 0:
            postings = getPostings_C0(getTokensFromText(line))
        elif compression == 1:
            postings = getPostings_C1(getTokensFromText(line))
        elif compression == 2:
            postings = getPostings_C2(getTokensFromText(line))
        elif compression == 3:
            postings = getPostings_C3(getTokensFromText(line))
        elif compression == 4:
            postings = getPostings_C4(getTokensFromText(line))

        docids = getIntersection(postings)
        for doc in docids:
            docno = int_to_docID[str(doc)]
            l = "Q"+str(qid) + " " + docno + " 1.0\n"
            res_file.write(l)
    res_file.close()
    queries.close()
    f.close()
