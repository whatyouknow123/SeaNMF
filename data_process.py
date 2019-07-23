'''
Visualize Topics
'''
import argparse
import jieba.posseg as pos

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--text_file', default='data/data.txt', help='input text file')
parser.add_argument('--corpus_file', default='data/doc_term_mat.txt', help='term document matrix file')
parser.add_argument('--corpus_origin_file', default='data/data_doc_term.txt', help='data doc term')
parser.add_argument('--vocab_file', default='data/vocab.txt', help='vocab file')
parser.add_argument('--vocab_max_size', type=int, default=10000, help='maximum vocabulary size')
parser.add_argument('--vocab_min_count', type=int, default=3, help='minimum frequency of the words')
parser.add_argument('--stopwords', default='data/stopwords', help='input stopwords file')
args = parser.parse_args()

# read stop words file
print("read stop words")
stopwords = set()
with open(args.stopwords, "r") as data:
    for word in data:
        stopwords.add(word.strip("\n"))
print("finish read stop words: ", len(stopwords))

# create vocabulary
print('create vocab')
vocab = {}
fp = open(args.text_file, 'r')
for line in fp:
    values = pos.cut(line.strip())
    arr = []
    for token, token_pos in values:
        if token_pos.find("n") != -1 and token not in stopwords:
            arr.append(token)
    for wd in arr:
        try:
            vocab[wd] += 1
        except:
            vocab[wd] = 1
print("finish read vocab: ", len(vocab))
fp.close()


vocab_arr = [[wd, vocab[wd]] for wd in vocab if vocab[wd] > args.vocab_min_count]
vocab_arr = sorted(vocab_arr, key=lambda k: k[1])[::-1]
vocab_arr = vocab_arr[:args.vocab_max_size]
vocab_arr = sorted(vocab_arr)

fout = open(args.vocab_file, 'w')
for itm in vocab_arr:
    itm[1] = str(itm[1])
    fout.write(' '.join(itm)+'\n')
fout.close()

# vocabulary to id
vocab2id = {itm[1][0]: itm[0] for itm in enumerate(vocab_arr)}
print('create document term matrix')
data_arr = []
fp = open(args.text_file, 'r')
fout = open(args.corpus_file, 'w')
fout2 = open(args.corpus_origin_file, "w")
sum = 0
count = 0
for line in fp:
    values = pos.cut(line.strip())
    arr = []
    for token, token_pos in values:
        if token_pos.find("n") != -1 and token not in stopwords:
            arr.append(token)
    arr_ids = [str(vocab2id[wd]) for wd in arr if wd in vocab2id]
    if len(arr_ids) > 0:
        sen = ' '.join(arr_ids)
        fout.write(sen+'\n')
        sen = " ".join(arr)
        fout2.write(line)
        sum += len(arr_ids)
        count += 1
fp.close()
fout.close()
fout2.close()
avg = sum / count if count > 0 else 0
print("sentence avg lenght is: ", avg)