import jieba.posseg as pos
import re
import numpy as np

# read stop words file
print("read stop words")
stopwords = set()
with open("data/stopwords", "r") as data:
    for word in data:
        stopwords.add(word.strip("\n"))
print("finish read stop words: ", len(stopwords))


def read_docs(file_name):
    print('read documents')
    print('-'*50)
    docs = []
    fp = open(file_name, 'r')
    for line in fp:
        arr = re.split('\s', line[:-1])
        arr = filter(None, arr)
        arr = [int(idx) for idx in arr]
        docs.append(arr)
    fp.close()
    
    return docs

def read_vocab(file_name):
    print('read vocabulary')
    print('-'*50)
    vocab = []
    fp = open(file_name, 'r')
    for line in fp:
        arr = re.split('\s', line[:-1])
        vocab.append(arr[0])
    fp.close()

    return vocab

def create_doc_term_mat(vocab_file, file_name, file_out):
    vocab_arr = read_vocab(vocab_file)
    # vocabulary to id
    vocab2id = {itm[1]: itm[0] for itm in enumerate(vocab_arr)}
    fout = open(file_out, "w")
    with open(file_name, "r") as data:
        for line in data:
            values = pos.cut(line.strip())
            arr = []
            for token, token_pos in values:
                if token_pos.find("n") != -1 and token not in stopwords:
                    arr.append(token)
            arr = [str(vocab2id[wd]) for wd in arr if wd in vocab2id]
            if len(arr) > 0:
                sen = ' '.join(arr)
                fout.write(sen + '\n')
    fout.close()


def calculate_PMI(AA, topKeywordsIndex):
    '''
    Reference:
    Short and Sparse Text Topic Modeling via Self-Aggregation
    '''
    D1 = np.sum(AA)
    n_tp = len(topKeywordsIndex)
    PMI = []
    for index1 in topKeywordsIndex:
        for index2 in topKeywordsIndex:
            if index2 < index1:
                if AA[index1, index2] == 0:
                    PMI.append(0.0)
                else:
                    C1 = np.sum(AA[index1])
                    C2 = np.sum(AA[index2])
                    PMI.append(np.log(AA[index1,index2]*D1/C1/C2))
    avg_PMI = 2.0*np.sum(PMI)/float(n_tp)/(float(n_tp)-1.0)

    return avg_PMI


if __name__ == "__main__":
    create_doc_term_mat("data/vocab.txt", "data/tianmaojingling_test_data_answer", "tianmaojingling_test")
