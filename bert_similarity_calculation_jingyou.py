#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import codecs
import pickle
import os
from datetime import timedelta, datetime
from run_ner import create_model, InputFeatures, InputExample
from bert import tokenization
from bert import modeling
import os
# import jieba
import pymongo
from ac_ahocorasick import ac_automation
# import ahocorasick
# import math
import re
import cx_Oracle
import pandas as pd
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
# import jieba.posseg as pesg
# from pyltp import Segmentorpyltp
import pkuseg
from bert_serving.client import BertClient

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "do_predict_outline", False,
    "Whether to do predict outline."
)
flags.DEFINE_bool(
    "do_predict_online", True,
    "Whether to do predict online."
)

# init mode and session
# move something codes outside of function, so that this code will run only once during online prediction when predict_online is invoked.
is_training=False
use_one_hot_embeddings=False
batch_size=1

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess=tf.Session(config=gpu_config)
model=None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None
print(FLAGS.output_dir)
print('checkpoint path:{}'.format(os.path.join(FLAGS.output_dir, "checkpoint")))
if not os.path.exists(os.path.join(FLAGS.output_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

# 加载label->id的词典
with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with codecs.open(os.path.join(FLAGS.output_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = len(label_list) + 1

graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="input_mask")
    label_ids_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="label_ids")
    segment_ids_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="segment_ids")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    (total_loss, logits, trans, pred_ids) = create_model(
        bert_config, is_training, input_ids_p, input_mask_p, segment_ids_p,
        label_ids_p, num_labels, use_one_hot_embeddings)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.output_dir))


tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


def predict_online(info):
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line):
        feature = convert_single_example(0, line, label_list, FLAGS.max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids],(batch_size, FLAGS.max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, FLAGS.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, FLAGS.max_seq_length))
        label_ids =np.reshape([feature.label_ids],(batch_size, FLAGS.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        # print(id2label)
        sentence = str(info)
        start = datetime.now()
        if len(sentence) < 2:
            print(sentence)
        sentence = tokenizer.tokenize(sentence)
        # print('your input is:{}'.format(sentence))
        input_ids, input_mask, segment_ids, label_ids = convert(sentence)

        feed_dict = {input_ids_p: input_ids,
                     input_mask_p: input_mask,
                     segment_ids_p:segment_ids,
                     label_ids_p:label_ids}
        # run session get current feed_dict result
        pred_ids_result = sess.run([pred_ids], feed_dict)
        pred_label_result = convert_id_to_label(pred_ids_result, id2label)
        # print(pred_label_result)
        #todo: 组合策略
        chexi,banxing = strage_combined_link_org_loc(sentence, pred_label_result[0], True)
        # result=''.join(result).replace(',','')
        # print('识别的实体有：{}'.format(''.join(result)))
        # print('Time used: {} sec'.format((datetime.now() - start).seconds))
        return chexi,banxing

def predict_outline():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line):
        feature = convert_single_example(0, line, label_list, FLAGS.max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids],(batch_size, FLAGS.max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, FLAGS.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, FLAGS.max_seq_length))
        label_ids =np.reshape([feature.label_ids],(batch_size, FLAGS.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        start = datetime.now()
        nlpcc_test_data = pd.read_csv("./Data/CHEXI_NER_Data/df_negtive.csv")
        correct = 0
        test_size = nlpcc_test_data.shape[0]
        nlpcc_test_result = []

        for row in nlpcc_test_data.index:
            info = nlpcc_test_data.loc[row,"MODEL_STANDARD_NAME"]
            entity = nlpcc_test_data.loc[row,"CATE_NAME"]
            # attribute = nlpcc_test_data.loc[row, "t_str"].split("|||")[1].strip()
            # answer = nlpcc_test_data.loc[row, "t_str"].split("|||")[2].strip()

            sentence = str(info)
            start = datetime.now()
            if len(sentence) < 2:
                print(sentence)
                continue
            sentence = tokenizer.tokenize(sentence)
            input_ids, input_mask, segment_ids, label_ids = convert(sentence)

            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask,
                         segment_ids_p:segment_ids,
                         label_ids_p:label_ids}
            # run session get current feed_dict result
            pred_ids_result = sess.run([pred_ids], feed_dict)
            pred_label_result = convert_id_to_label(pred_ids_result, id2label)
            # print(pred_label_result)

            result = strage_combined_link_org_loc(sentence, pred_label_result[0], False)
            if entity in result:
                correct += 1
                print(correct)
            else:
                print(str(info)+"\t"+str(entity)+"\t"+str(','.join(result)).replace(',',''))
            nlpcc_test_result.append(str(info)+"\t"+str(entity)+"\t"+str(','.join(result)).replace(',',''))
        print("accuracy: {}%, correct: {}, total: {}".format(correct * 100.0 / float(test_size), correct, test_size))
        print('Time used: {} sec'.format((datetime.now() - start).seconds))
        with open("./Data/CHEXI_NER_Data/negtive_predict.txt", "w",encoding='utf-8') as f:
            f.write("\n".join(nlpcc_test_result))



def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result


def strage_combined_link_org_loc(tokens, tags, flag):
    """
    组合策略
    :param pred_label_result:
    :param types:
    :return:
    """
    def print_output(data, type):
        line = []
        for i in data:
            line.append(i.word)
        return  type, line
        # print('{}: {}'.format(type, ', '.join(line)))

    def string_output(data):
        line = []
        for i in data:
            line.append(i.word)
        return line

    params = None
    eval = Result(params)
    if len(tokens) > len(tags):
        tokens = tokens[:len(tags)]
    person, loc, org = eval.get_result(tokens, tags)
    if flag:
        if len(loc) != 0:
            loc_type,chexi=print_output(loc, 'LOC')
        else:
            chexi =[]
        if len(person) != 0:
            print_output(person, 'PER')
        if len(org) != 0:
            org_type,banxing=print_output(org, 'ORG')
        else:
            banxing =[]
    return chexi, banxing
    # person_list = string_output(person)
    # person_list.extend(string_output(loc))
    # person_list.extend(string_output(org))



def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(FLAGS.output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


class Pair(object):
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start
    @property
    def end(self):
        return self.__end
    @property
    def merge(self):
        return self.__merge
    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types
    @word.setter
    def word(self, word):
        self.__word = word
    @start.setter
    def start(self, start):
        self.__start = start
    @end.setter
    def end(self, end):
        self.__end = end
    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type



class Result(object):
    def __init__(self, config):
        self.config = config
        self.person = []
        self.loc = []
        self.org = []
        self.others = []
    def get_result(self, tokens, tags, config=None):
        # 先获取标注结果
        self.result_to_json(tokens, tags)
        return self.person, self.loc, self.org

    def result_to_json(self, string, tags):
        """
        将模型标注序列和输入序列结合 转化为结果
        :param string: 输入序列
        :param tags: 标注结果
        :return:
        """
        item = {"entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        last_tag = ''

        for char, tag in zip(string, tags):
            if tag[0] == "S":
                self.append(char, idx, idx+1, tag[2:])
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "O":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
            last_tag = tag
        if entity_name != '':
            self.append(entity_name, entity_start, idx, last_tag[2:])
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
        return item

    def append(self, word, start, end, tag):
        if tag == 'LOC':
            self.loc.append(Pair(word, start, end, 'LOC'))
        elif tag == 'PER':
            self.person.append(Pair(word, start, end, 'PER'))
        elif tag == 'ORG':
            self.org.append(Pair(word, start, end, 'ORG'))
        else:
            self.others.append(Pair(word, start, end, tag))


class SimWordVec:
    def __init__(self):
        self.bc = BertClient()
        # self.embedding_path = 'model/skipgram_wordvec1.bin'
        # self.model = gensim.models.KeyedVectors.load_word2vec_format(self.embedding_path, binary=False)
    '''获取词向量'''
    def get_wordvector(self, word):#获取词向量
        try:
            return self.model[word]
        except:
            return np.zeros(300)
    '''基于余弦相似度计算句子之间的相似度，句子向量等于字符向量求平均'''
    def similarity_cosine(self, word_list1,word_list2):#给予余弦相似度的相似度计算
        vector1 = self.bc.encode(word_list1)[0]
        vector2=self.bc.encode(word_list2)[0]
        cos1 = np.sum(vector1*vector2)
        cos21 = np.sqrt(sum(vector1**2))
        cos22 = np.sqrt(sum(vector2**2))
        similarity = cos1/float(cos21*cos22)
        return  similarity
    '''计算句子相似度'''
    def distance(self, text1, text2):#相似性计算主函数
        word_list1=[word for word in list(text1) if word]
        word_list1=[''.join(word_list1)]
        word_list2=[word for word in list(text2) if word]
        word_list2 = [''.join(word_list2)]
        return self.similarity_cosine(word_list1,word_list2)

class Doc_most_sim():
    def __init__(self):
        CUR_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # LTP_DIR = "D:\code\ltp_data_v3.4.0"
        DICT_DIR = os.path.join(CUR_DIR, 'data1')
        # LoaddictPath=os.path.join(DICT_DIR, 'dict.txt')
        # self.DictPath = os.path.join(DICT_DIR, 'car_series_jingyou - 副本.txt')
        self.ChexiPath=os.path.join(DICT_DIR, 'car_series_jingyou.txt')
        self.SynonymPath = os.path.join(DICT_DIR, '车系同义词.txt')
        self.BanXingpath = os.path.join(DICT_DIR, 'car_series_jingyou_banxing.txt')
        self.Biansupath=os.path.join(DICT_DIR, 'car_series_jingyou_biansuxiang.txt')
        # self.PaiLiangpath=os.path.join(DICT_DIR,'car_series_jingyou_pailiang.txt')
        # self.EcList=os.path.join(DICT_DIR, '')
        # YearPath=os.path.join(DICT_DIR, 'year.txt')
        # ContentPath=os.path.join(DICT_DIR, 'content.txt')
        # self.segmentor = Segmentor()
        # self.segmentor.load(os.path.join(LTP_DIR, "cws.model"))
        # self.segmentor.load_with_lexicon(os.path.join(LTP_DIR, "cws.model"), CarnumberPath)  # 加载模型，第二个参数是您的外部词典文件路径
        self.ChexiList = [i.strip() for i in open(self.ChexiPath,encoding='utf-8') if i.strip()]
        self.SynonymDict = {i.strip().split(';')[0]:i.strip().split(';')[1] for i in open(self.SynonymPath,encoding='utf-8') if i.strip()}
        self.BanXingList=[i.strip() for i in open(self.BanXingpath,encoding='utf-8') if i.strip()]
        self.BiansuList=[i.strip() for i in open(self.Biansupath,encoding='utf-8') if i.strip()]
        # self.PaiLiangList=[i.strip() for i in open(self.PaiLiangpath,encoding='utf-8') if i.strip()]
        # self.YearList = [i.strip() for i in open(YearPath,encoding='utf-8') if i.strip()]

        # self.ContentList=[i.strip() for i in open(ContentPath,encoding='utf-8') if len(i)>0]
        # self.ChexiTree = self.build_actree(self.ChexiList)
        # self.BiansuTree = self.build_actree(self.BiansuList)
        # self.BanXingTree=self.build_actree(self.BanXingList)
        # self.PaiLiangTree=self.build_actree(self.PaiLiangList)
        # self.Ectree=self.build_actree(self.EcList)
        # self.YearTree = self.build_actree(self.YearList)
        # self.UserWords = list(set(self.DictPath))
        # jieba.load_userdict(self.DictPath)
        self.simer = SimWordVec()
        self.seg = pkuseg.pkuseg(user_dict=self.get_list_cars())
        conn = pymongo.MongoClient()
        self.db = conn['XS_CAR']['data']
        self.status='None'
    def get_list_cars(self):
        dict_cars = []
        file = open("./data1/car_series_jingyou_all.txt",encoding='utf-8')
        for line in file:
            dict_cars.append(line.strip('\n'))
        file.close()
        return  dict_cars
    def getData(self,user, password, database, targetTable, commandText):
        # start_time = datetime.datetime.now()
        connection = cx_Oracle.connect(user, password, database)
        cursor = connection.cursor()
        cursor.execute(commandText.format(targetTable))
        x = cursor.description
        columns = [y[0] for y in x]
        cursor01 = cursor.fetchall()
        cursor.close()
        data = pd.DataFrame(cursor01, columns=columns)
        # end_time = datetime.datetime.now()
        # ss = str((end_time - start_time).seconds)
        # print(ss)
        return data

    def build_actree(self, wordlist):
        actree = ac_automation()
        for index, word in enumerate(wordlist):
            # word = ' ' + word + ' '
            actree.add(word)
        return actree

    '''content分句处理'''

    def seg_sentences(self, content):
        return [content.replace('\u3000', '').replace('\xc2\xa0', '').replace(' ','')]

    '''过滤'''

    def check_chexi(self, sentence):
        flag = 0
        word_list = self.seg.cut(sentence)
        # word_list = list(jieba.cut(sentence))
        # print(word_list)
        senti_words = []
        for i in self.ChexiTree.search(' '.join(word_list)):
            senti_words.append(i.replace(' ', ''))
            flag += 1
        return flag, word_list, senti_words

    def check_banxing(self, sentence):
        flag = 0
        word_list = self.seg.cut(sentence)
        # word_list = list(jieba.cut(sentence))
        # print(word_list)
        senti_words = []
        for i in self.BanXingTree.search(' '.join(word_list)):
            senti_words.append(i.replace(' ', ''))
            flag += 1
        return flag, word_list, senti_words

    def check_biansu(self, sentence):
        flag = 0
        word_list = self.seg.cut(sentence)
        # word_list = list(jieba.cut(sentence))
        # print(word_list)
        senti_words = []
        for i in self.BiansuTree.search(' '.join(word_list)):
            senti_words.append(i.replace(' ', ''))
            flag += 1
        return flag, word_list, senti_words
    def pailiang_standard(self,sentence):
        '''对排量标准进行标准化,查找排量标准'''
        sentence=sentence.replace('-','').replace('+','').replace('(','').replace(')','').replace('）','').replace('（','').replace('Ⅳ','IV').replace('Ⅴ','V').replace('Ⅵ','VI').replace('Ⅲ','III').replace('国五','国V').replace('国二','国II')
        pailiang=re.findall('([国欧京][IV]{1,4}|OBD)', sentence)
        return sentence, pailiang
    def pailiang_judge(self,pailiang_list,content_pailiang_list):
        '''判断排量标准是否包含全部包含返回True否则返回False'''
        count=0
        for pl in pailiang_list:
            if pl in list(set(content_pailiang_list)):
                count+=1
            else:
                continue
        if count==len(list(set(content_pailiang_list))):
            return True
        else:
            return False
    def banxing_judge(self,senti_words_banxing,content):
        '''判断版型是否包含全部包含返回True否则返回False'''
        count=0
        for pl in senti_words_banxing:
            if pl in content:
                count+=1
            else:
                continue
        if count>=1:
            return True
        else:
            return False
    def export_data_list(self,commandText):
        i = 0
        list1=[]
        for item in self.db.find(commandText):
            i += 1
            STD_MODEL_TOGETHER_ADD=item['STD_MODEL_CONCAT']
            list1.append(STD_MODEL_TOGETHER_ADD)
        return list1
    def synonym_exchange(self,sentence):
        synonym_dict = {}
        synonym = [i.strip() for i in open('./data/车系dict.txt') if i.strip()]
        for s in synonym:
            synonym_dict[s.split(';')[0]] = s.split(';')[1]
        for sd in synonym_dict:
            if sd in sentence:
                sentence=sentence.replace(sd,synonym_dict[sd])
        return sentence
    # def check_ec(self, sentence):
    #     flag = 0
    #     word_list = self.seg.cut(sentence)
    #     # word_list = list(jieba.cut(sentence))
    #     # print(word_list)
    #     senti_words = []
    #     for i in self.BrandTree.search(' '.join(word_list)):
    #         senti_words.append(i.replace(' ', ''))
    #         flag += 1
    #     return flag, word_list, senti_words
    '''正则精准匹配车系'''
    def re_match_chexi(self,sentence):
        try:
            compil = '|'.join(self.ChexiList)
            chexi = re.search(compil, sentence).group()
        except:
            chexi =''
        return chexi

    def re_match_biansu(self, sentence):
        try:
            compil = '|'.join(self.BiansuList)
            biansu = re.findall(compil, sentence)
        except:
            biansu =[]
        return biansu

    def re_match_banxing(self,sentence):
        try:
            compil = '|'.join(self.BanXingList)
            banxing= re.findall(compil, sentence)
        except:
            banxing =[]
        return banxing

    '''提取最相似内容 车系，年款，排量，长宽高，变速箱，版型'''

    def extract_most_sim_content_all(self, sentences,ff):
        #ba = self.get_list_cars()
        sims = []
        # user = 'dd_data'
        # password = 'xdf123'
        # database = 'LBORA170'
        # targetTable = 'soure_01'
        chexis=[]
        banxings=[]
        for index, sentence in enumerate(sentences):
            sentence=sentence.upper()
            sentence,pailiang_list=self.pailiang_standard(sentence)
            try:
                lwh = str(re.search('(\d{0,4}\*\d{0,4}\*\d{0,4})', sentence).group())
            except:
                lwh=''
            if len(lwh)<12:
                lwh=''
            chexi,banxing=predict_online(sentence)
            if len(chexi):
                for cx in chexi:
                    chexis.append(cx)
            if len(banxing):
                for bx in banxing:
                    banxings.append(bx)
            chexi_re=self.re_match_chexi(sentence)
            if len(chexi_re):
                chexis.append(chexi_re)
            senti_words=list(set(chexis))
            banxing_re=self.re_match_banxing(sentence)
            banxings.extend(banxing_re)
            if banxings==[]:
                banxings=['']
            if len(senti_words):
                id2content = {}
                sim2id = {}
                for k in range(len(senti_words)):
                    car_system=senti_words[k]
                    if car_system in self.SynonymDict:
                        chexi=self.SynonymDict[car_system]
                    else:
                        chexi=car_system
                    value = '{}'.format(chexi)
                    commandText = {'FIRST_CARS_QUKONG': {'$regex': value}}
                    cs_cb_data = self.export_data_list(commandText)
                    if len(cs_cb_data):
                        try:
                            year = str(re.search('.*(19\d{2}|20\d{2})[款改 ]', sentence).group(1))
                            try:
                                fuel_consumption = str(re.search('.*?(\d{1}.\d{1})[TL]', sentence).group(1))
                            except:
                                fuel_consumption = ''
                            # seats = str(re.search('(\d{1}座)', sentence))

                            try:
                                engine = re.search('(?<!\d)\d\d(?!\d)', sentence).group()
                            except:
                                engine =''
                            count = 0
                            biansu = self.re_match_biansu(sentence)
                            if biansu==[]:
                                biansu=['']
                            for bs in biansu:
                                for j in range(len(cs_cb_data)):
                                    # cs_cb_data['STD_MODEL_TOGETHER_ADD']
                                    # content = str(cs_cb_data['STD_MODEL_TOGETHER_ADD'][j])
                                    content = str(cs_cb_data[j])
                                    content = content.upper()
                                    cn = content.upper().replace(' ', '').replace('\r', '').replace('\u3000', '')
                                    if len(banxings):
                                        for bx in banxings:
                                            if year in content and fuel_consumption in content and bx in content and lwh in content\
                                                    and bs in content and engine in cn[:20]:
                                                id2content[count] = content
                                                sim = (self.simer.distance(sentence, cn))
                                                # print(sim)
                                                if 'PLUS' in content and 'PLUS' in sentence:
                                                    sim = sim + 0.01
                                                elif 'PLUS' in content and 'PLUS' not in sentence:
                                                    sim = sim - 0.01
                                                sim2id[sim] = count
                                                sims.append(sim)
                                                count += 1
                                                self.status = '1'
                                            elif '电动' in sentence and '电动' in content and year in content and lwh in content\
                                                 and bx in content:
                                                id2content[count] = content
                                                sim = (self.simer.distance(sentence, cn))
                                                # print(sim)
                                                if 'PLUS' in content and 'PLUS' in sentence:
                                                    sim = sim + 0.01
                                                elif 'PLUS' in content and 'PLUS' not in sentence:
                                                    sim = sim - 0.01
                                                sim2id[sim] = count
                                                sims.append(sim)
                                                count += 1
                                                self.status = '2'
                                            # elif year in content and fuel_consumption in content and bx in content and lwh in content \
                                            #         and bs in content:
                                            #     id2content[count] = content
                                            #     sim = (self.simer.distance(sentence, cn))
                                            #     # print(sim)
                                            #     if 'PLUS' in content and 'PLUS' in sentence:
                                            #         sim = sim + 0.01
                                            #     elif 'PLUS' in content and 'PLUS' not in sentence:
                                            #         sim = sim - 0.01
                                            #     sim2id[sim] = count
                                            #     sims.append(sim)
                                            #     count += 1
                                            #     self.status = '2'
                                            # elif '电动' in sentence and '电动' in content and year in content and lwh in content \
                                            #         and bx in content:
                                            #     id2content[count] = content
                                            #     sim = (self.simer.distance(sentence, cn))
                                            #     # print(sim)
                                            #     if 'PLUS' in content and 'PLUS' in sentence:
                                            #         sim = sim + 0.01
                                            #     elif 'PLUS' in content and 'PLUS' not in sentence:
                                            #         sim = sim - 0.01
                                            #     sim2id[sim] = count
                                            #     sims.append(sim)
                                            #     count += 1
                                            #     self.status = '2'
                                            else:
                                                continue
                                                self.status = 'None'

                        except Exception as e:
                            print(e)
                    else:
                        continue

        try:
            #sims=[round(r,2) for r in sims]
            max_sim = max(sims)
            sim_w = round(max_sim, 2)
            if sim_w > 0.80:
                result = id2content[sim2id[max_sim]]
                return result+'CUTLINE'+self.status
            else:
                return '最大相似度小于阈值'

        except Exception as e:
            print(e)
            return '没有这个年款或没有匹配到车系'






    def getData1(self,commmat):
        connection = cx_Oracle.connect("dd_data", "xdf123", "LBORA170")
        cursor = connection.cursor()
        cursor.execute(commmat)
        x = cursor.description
        columns = [y[0] for y in x]
        cursor01 = cursor.fetchall()
        cursor.close()
        data = pd.DataFrame(cursor01, columns=columns)


        return data

    def upData(self,commat):
        connection = cx_Oracle.connect("dd_data", "xdf123", "LBORA170")
        cursor = connection.cursor()
        cursor.execute(commat)
        connection.commit()
        cursor.close()
        connection.close()
        return

    '''根据条件进行筛选，条件，车系，年款，排量，长宽高，变速箱，版型'''
    def doc_score_all(self, content):
        sents = self.seg_sentences(content)
        result_content = self.extract_most_sim_content_all(sents,content)
        return result_content
if __name__=='__main__':
    import time
    def get_time_dif(start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))
    start=time.time()
    handle=Doc_most_sim()
    with open('data1/jingyou_cxpp_out.csv','w',encoding='utf-8') as writer:
        # results=[]
        for i,line in enumerate(open('data1/jingyou_cxpp.csv',encoding='utf-8')):
            content =line.strip().upper()
            #print(content)
            start_time = datetime.now()
            result=handle.doc_score_all(content)
            writer.write(content + 'CUTLINE' + result+ '\n')
            # results.append(str(result))
            end_time = datetime.now()
            s=str(i)+'\t'+str((end_time - start_time).seconds)
            print(s)
    print(get_time_dif(start))



    # for line in ['一汽奥迪 奥迪Q3 2017款 35 TFSI 2.0T 双离合 两驱 运动型前置前驱 4398*1841*1591 DCT 5门5座2.0T国IV(国V)']:
    #     content =line.strip()
    #     start_time = datetime.now()
    #     content=content.upper()
    #     result=handle.doc_score_all(content)
    #     end_time = datetime.now()
    #     s = str((end_time - start_time).seconds)
    #     print(s)
    #     print(result)
    # print(get_time_dif(start))