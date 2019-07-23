'''
Visualize Topics
'''
from utils import *
import argparse
import synonyms
from collections import defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from numpy.linalg import norm
import numpy.linalg as lg



parser = argparse.ArgumentParser()
parser.add_argument('--corpus_file', default='data/doc_term_mat.txt', help='term document matrix file')
parser.add_argument('--vocab_file', default='data/vocab.txt', help='vocab file')
parser.add_argument('--par_file', default='seanmf_results/W.txt', help='model weight results file')
parser.add_argument('--predict_file', default='data/predict.txt', help='predict file')
parser.add_argument('--height_file', default='seanmf_results/H.txt', help="model height results file")
opt = parser.parse_args()

wc = WordCloud(font_path="/Users/wuyanjing/code/alibaba/voice-insight-alg/topic_dection/data/topic_generation/fonts/simkai.ttf",  # 设置字体
               background_color="white",  # 背景颜色
               max_words=2000,  # 词云显示的最大词数
               max_font_size=100,  # 字体最大值
               random_state=42,
               width=1000, height=860, margin=2,# 设置图片默认的大小,但是如果使用背景图片的话,那么保存的图片大小将会按照其大小保存,margin为词语边缘距离
               )

docs = read_docs(opt.corpus_file)
vocab = read_vocab(opt.vocab_file)
vocab2id = {itm[1]: itm[0] for itm in enumerate(vocab)}
n_docs = len(docs)
n_terms = len(vocab)
print('n_docs={}, n_terms={}'.format(n_docs, n_terms))

dt_mat = np.zeros([n_terms, n_terms])
for itm in docs:
    for kk in itm:
        for jj in itm:
            if kk != jj:
                dt_mat[int(kk), int(jj)] += 1.0
print('co-occur done')
        
W = np.loadtxt(opt.par_file, dtype=float)
H = np.loadtxt(opt.height_file, dtype=float)
n_topic = W.shape[1]
W /= W.max(axis=0)
print('n_topic={}'.format(n_topic))

PMI_arr = []
n_topKeyword = 10
for k in range(n_topic):
    topKeywordsIndex = W[:,k].argsort()[::-1][:n_topKeyword]
    PMI_arr.append(calculate_PMI(dt_mat, topKeywordsIndex))
print('Average PMI={}'.format(np.average(np.array(PMI_arr))))

index = np.argsort(PMI_arr)


def create_wordcloud(show_data, topic_num):
    wc.generate_from_frequencies(show_data)
    plt.figure()
    # 以下代码显示图片
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
    # 绘制词云

    # 保存图片
    wc.to_file("image/" + topic_num + ".png")

def show_topic():
    topics = []
    for k in range(n_topic):
        '''
        print('Topic ' + str(k+1) + ': ')
        print(PMI_arr[k])

        keyword_weight = defaultdict(int)
        for w in np.argsort(W[:,k])[::-1][:n_topKeyword]:
            keyword_weight[vocab[w]] = 1
            for m in np.argsort(W[:,k])[::-1][:n_topKeyword]:
                similarity = 0
                try:
                    similarity = synonyms.compare(vocab[w], vocab[k])
                except:
                    pass
                if similarity > 0.8:
                    keyword_weight[vocab[m]] += 1
        '''
        max_topic_index = np.argmax(W[:, k])
        topics.append(vocab[max_topic_index])
        #for w in np.argmax(W[:,k]):
            #topics.append(vocab[w])
            #print("topic:", vocab[w])
            #print("topic weight: ", W[w, :])
            #print("topic weight index:", np.argsort(-W[w, :])[0])
        #count += 1
    print("topis: ", topics)

    return topics

def tag_label(path):
    labels = show_topic()
    save_file = open(path, "w")
    count = 0
    none = 0
    token_correct = 0
    label_map = {'音质': ["音质", "声音", "低音", "杂音", "音效", "音量"], "续航": ["续航", "待机时间"], "蓄电": ["电池", "电池容量", "电"], "便携": ["外带", "便携性"],
                 "防水": [], "颜色": ["颜色", "红色"], "外形": ["外观", "设计"],
                 "笨重": ["体积", "太重"], "连接距离": ["距离"], "音乐内容": ["音乐", "资源", "内容", "歌曲", "音乐品味", "歌"],
                 "功能": ["功能", "智能设备语音控制功能", "语音搜索功能", "收款功能", "待办事项提醒功能", "百科功能", "闲聊功能", "购物功能",  "话费充值功能",
                        "外卖功能", "语音留言功能", "话费", "语音", "闹钟功能", "闹钟", "找手机功能", "声纹识别", "自定义问答功能"],
                 "稳定性": ["联网稳定性"], "价格": ["价格", "太贵", "性价比", "一分货", "小贵", "实惠"], "质量": ["质量", "质感", "材质"],
                 "连接问题": ["蓝牙", "插卡", "蓝牙连接功能"], "无": ["None"], "联网": ["联网", "传输", "联网稳定性"],
                 "智能": ["智能", "对话", "自动", "对话流畅度", "Ai"], "收音机": ["收音机"], "接口": [], "表意不明": ["None"],
                 "其他": [], "按键": ["按键"]}
    label_conv_map = {}
    for topic in label_map:
        for extend_topic in label_map[topic]:
            label_conv_map[extend_topic] = topic
    with open(opt.predict_file, "r") as data:
        for line in data:
            parts = line.strip().split("\t")
            answer = ""
            label = "None"
            if len(parts) == 2:
                answer = parts[0]
                label = parts[1]
            elif len(parts) == 1:
                answer = parts[0]
            else:
                continue
            if label == "表意不明" or label == "None" or label == "其它":
                continue
            data_label = label.split("|")
            values = pos.cut(answer.strip())
            arr = []
            BOW = np.zeros([n_terms, 1])
            for token, token_pos in values:
                if token_pos.find("n") != -1 and token not in stopwords:
                    arr.append(token)
            tokens = [int(vocab2id[wd]) for wd in arr if wd in vocab2id]
            if len(tokens) <= 0: continue
            for token in tokens:
                BOW[token][0] += 1
            topics = np.zeros([1, n_topic])
            token_label = set()
            if len(tokens) > 0:
                for token in tokens:
                    topics += W[token, :]
                    if np.max(W[token, :]) > 0.95:
                        token_label.add(labels[np.argsort(-W[token, :])[0]])
                topics /= norm(topics)
            for each in token_label:
                current_label = each
                if current_label in label_conv_map:
                    current_label = label_conv_map[current_label]
                if current_label in data_label:
                    token_correct += 1
                    break
            if not len(token_label):
                none += 1
                token_label.add("None")
            token_labels = "|".join(token_label)
            
            save_file.write("%s\t%s\n" % (line.strip(), token_labels))
            count += 1
        print("token correct: ", token_correct * 1.0 / count)
        print("token cover ratio: ", (count - none) * 1.0 / count)
    save_file.close()


def tag_H_label(path):
    h_correct = 0
    labels = show_topic()
    save_file = open(path, "w")
    count = 0
    with open(opt.predict_file, "r") as data:
        for line in data:
            parts = line.strip().split("\t")
            answer = ""
            label = "None"
            if len(parts) == 2:
                answer = parts[0]
                label = parts[1]
            elif len(parts) == 1:
                answer = parts[0]
            else:
                continue
            data_label = label.split("|")

            current_label = []
            for index in range(H[count, :].shape[0]):
                if H[count, index] > 0.8:
                    current_label.append(labels[index])
            for each in current_label:
                if each in data_label:
                    h_correct += 1
                    break
            if len(current_label) == 0:
                current_labels = "None"
            else:
                current_labels = "|".join(current_label)
            save_file.write("%s\t%s\n" % (answer, current_labels))
            count += 1
            print("h_correct: ", h_correct * 1.0 / count)
    save_file.close()



if __name__ == "__main__":
    path2 = "data/tianmaojingling_H_test_label"
    #tag_H_label(path2)
    path = "data/tianmaojingling_test_label"
    tag_label(path)
    #show_topic()

