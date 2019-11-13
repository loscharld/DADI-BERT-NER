#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import jieba
import pymongo
# from ac_ahocorasick import ac_automation
# import ahocorasick
import math
import re
import cx_Oracle
import pandas as pd
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
import jieba.posseg as pesg
# from pyltp import Segmentorpyltp
import datetime
import pkuseg

class SimWordVec:
    def __init__(self):
        self.embedding_path = 'model/skipgram_wordvec1.bin'
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.embedding_path, binary=False)
    '''获取词向量'''
    def get_wordvector(self, word):#获取词向量
        try:
            return self.model[word]
        except:
            return np.zeros(300)
    '''基于余弦相似度计算句子之间的相似度，句子向量等于字符向量求平均'''
    def similarity_cosine(self, word_list1,word_list2):#给予余弦相似度的相似度计算
        vector1 = np.zeros(300)
        for word in word_list1:
            vector1 += self.get_wordvector(word)
        vector1=vector1/len(word_list1)
        vector2=np.zeros(300)
        for word in word_list2:
            vector2 += self.get_wordvector(word)
        vector2=vector2/len(word_list2)
        cos1 = np.sum(vector1*vector2)
        cos21 = np.sqrt(sum(vector1**2))
        cos22 = np.sqrt(sum(vector2**2))
        similarity = cos1/float(cos21*cos22)
        return  similarity
    '''计算句子相似度'''
    def distance(self, text1, text2):#相似性计算主函数
        word_list1=[word for word in text1 if len(word)<10]
        word_list2=[word for word in text2 if len(word)<10]
        return self.similarity_cosine(word_list1,word_list2)

class Doc_most_sim():
    def __init__(self):
        CUR_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # LTP_DIR = "D:\code\ltp_data_v3.4.0"
        DICT_DIR = os.path.join(CUR_DIR, 'data')
        # LoaddictPath=os.path.join(DICT_DIR, 'dict.txt')
        # self.DictPath = os.path.join(DICT_DIR, 'car_series_jingyou - 副本.txt')
        self.ChexiPath=os.path.join(DICT_DIR, 'car_series_jingyou.txt')
        self.BrandPath = os.path.join(DICT_DIR, 'brand.txt')
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
        self.BrandList = [i.strip() for i in open(self.BrandPath,encoding='utf-8') if i.strip()]
        self.BanXingList=[i.strip() for i in open(self.BanXingpath,encoding='utf-8') if i.strip()]
        self.BiansuList=[i.strip() for i in open(self.Biansupath,encoding='utf-8') if i.strip()]
        # self.PaiLiangList=[i.strip() for i in open(self.PaiLiangpath,encoding='utf-8') if i.strip()]
        # self.YearList = [i.strip() for i in open(YearPath,encoding='utf-8') if i.strip()]

        # self.ContentList=[i.strip() for i in open(ContentPath,encoding='utf-8') if len(i)>0]
        self.ChexiTree = self.build_actree(self.ChexiList)
        self.BiansuTree = self.build_actree(self.BiansuList)
        self.BanXingTree=self.build_actree(self.BanXingList)
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
        file = open("./data/car_series_jingyou_all.txt",encoding='utf-8')
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
        return [content.replace('\u3000', '').replace('\xc2\xa0', '')]

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


    '''提取最相似内容 车系，年款，排量，长宽高，变速箱，版型'''

    def extract_most_sim_content_all(self, sentences,ff):
        #ba = self.get_list_cars()
        sims = []
        # user = 'dd_data'
        # password = 'xdf123'
        # database = 'LBORA170'
        # targetTable = 'soure_01'
        for index, sentence in enumerate(sentences):
            sentence=self.synonym_exchange(sentence)
            sentence=sentence.upper()
            lwh = str(re.search('(\d{0,4}\*\d{0,4}\*\d{0,4})', sentence).group(1))
            sentence=sentence.replace('-','').replace('+','').replace('(','').replace(')','').replace('）','').replace('（','')
            sentence,pailiang_list=self.pailiang_standard(sentence)
            flag, word_list, senti_words = self.check_chexi(sentence)
            flag_banxing,word_list_banxing,senti_words_banxing=self.check_banxing(sentence)
            senti_words = list(set(senti_words))
            senti_words_banxing=list(set(senti_words_banxing))
            if flag:
                id2content = {}
                sim2id = {}
                for k in range(len(senti_words)):
                    # commandText1 = '''select t.std_model_together_add from XS_CAR_QUCHONG t where t.first_cars_qukong='{}' '''.format(
                    #     senti_words[k])
                    #cs_cb_data = self.getData(user, password, database, targetTable, commandText1)
                    value = '{}'.format(senti_words[k])
                    commandText = {'FIRST_CARS_QUKONG': {'$regex': value}}
                    cs_cb_data = self.export_data_list(commandText)
                    if len(cs_cb_data):
                        try:
                            year = str(re.search('.*(19\d{2}|20\d{2})[款 ]', sentence).group(1))
                            try:
                                fuel_consumption = str(re.search('.*?(\d{1}.\d{1})[TL]', sentence).group(1))
                            except:
                                fuel_consumption = ''
                            # seats = str(re.search('(\d{1}座)', sentence))
                            if senti_words_banxing==[]:
                                senti_words_banxing=['']
                            for p in range(len(senti_words_banxing)):
                                bangxing = senti_words_banxing[p]
                                count = 0
                                for j in range(len(cs_cb_data)):
                                    # cs_cb_data['STD_MODEL_TOGETHER_ADD']
                                    # content = str(cs_cb_data['STD_MODEL_TOGETHER_ADD'][j])
                                    content = str(cs_cb_data[j])
                                    content = content.upper()
                                    lwh_content = str(re.search('(\d{0,4}\*\d{0,4}\*\d{0,4})', content).group(1))
                                    # banxing_content, content_banixng_list = self.banixng_standard(content)
                                    pailiang_content, content_pailiang_list = self.pailiang_standard(content)
                                    # if senti_words_banxing!=[]:
                                    # for banxing in senti_words_banxing:
                                    # if banxing in content:
                                    flag_biansu, word_list_biansu, senti_words_biansu = self.check_biansu(sentence)
                                    flag_biansu_content, word_list_biansu_content, senti_words_biansu_content = self.check_biansu(
                                        content)
                                    try:
                                        if flag_biansu:
                                            biansu = senti_words_biansu[0]
                                    except:
                                        biansu = ''
                                    if year in content and fuel_consumption in content \
                                            and fuel_consumption != '' and lwh == lwh_content and biansu in senti_words_biansu_content \
                                            and bangxing in content and bangxing!='':
                                        # and bangxing in content and self.pailiang_judge(pailiang_list, content_pailiang_list) is True
                                        id2content[count] = content
                                        cn = self.seg.cut(
                                            content.upper().replace(' ', '').replace('\r', '').replace('\u3000', ''))
                                        # cn = list(jieba.cut(
                                        # content.upper().replace('\r', '').replace(' ', '').replace('\u3000', '')))
                                        sim = (self.simer.distance(word_list, cn))
                                        # print(sim)
                                        if 'PLUS' in content and 'PLUS' in sentence:
                                            sim = sim + 0.01
                                        elif 'PLUS' in content and 'PLUS' not in sentence:
                                            sim = sim - 0.01
                                        sim2id[sim] = count
                                        sims.append(sim)
                                        count += 1
                                        self.status = '1'
                                    elif '电动' in sentence and year in content \
                                            and lwh == lwh_content and bangxing in content and bangxing!='':
                                        # and bangxing in content and lwh == lwh_content
                                        id2content[count] = content
                                        cn = self.seg.cut(
                                            content.upper().replace(' ', '').replace('\r', '').replace('\u3000', ''))
                                        # cn = list(jieba.cut(
                                        # content.upper().replace('\r', '').replace(' ', '').replace('\u3000', '')))
                                        sim = (self.simer.distance(word_list, cn))
                                        if 'PLUS' in content and 'PLUS' in sentence:
                                            sim = sim + 0.01
                                        elif 'PLUS' in content and 'PLUS' not in sentence:
                                            sim = sim - 0.01
                                        sim2id[sim] = count
                                        sims.append(sim)
                                        count += 1
                                        self.status = '1'
                                    #   '''提取最相似内容 车系，年款，排量，长宽高，变速箱'''
                                    elif year in content and fuel_consumption in content \
                                            and fuel_consumption != '' and lwh == lwh_content and biansu in senti_words_biansu_content:
                                        # and bangxing in content and self.pailiang_judge(pailiang_list, content_pailiang_list) is True
                                        id2content[count] = content
                                        cn = self.seg.cut(
                                            content.upper().replace(' ', '').replace('\r', '').replace('\u3000',
                                                                                                       ''))
                                        # cn = list(jieba.cut(
                                        # content.upper().replace('\r', '').replace(' ', '').replace('\u3000', '')))
                                        sim = (self.simer.distance(word_list, cn))
                                        if 'PLUS' in content and 'PLUS' in sentence:
                                            sim += 0.01
                                        elif 'PLUS' in content and 'PLUS' not in sentence:
                                            sim = sim - 0.01
                                        # print(sim)
                                        sim2id[sim] = count
                                        sims.append(sim)
                                        count += 1
                                        self.status = '2'
                                    elif '电动' in sentence and year in content \
                                            and lwh == lwh_content:
                                        # and bangxing in content and lwh == lwh_content
                                        id2content[count] = content
                                        cn = self.seg.cut(
                                            content.upper().replace(' ', '').replace('\r', '').replace('\u3000',
                                                                                                       ''))
                                        # cn = list(jieba.cut(
                                        # content.upper().replace('\r', '').replace(' ', '').replace('\u3000', '')))
                                        sim = (self.simer.distance(word_list, cn))
                                        if 'PLUS' in content and 'PLUS' in sentence:
                                            sim += 0.01
                                        elif 'PLUS' in content and 'PLUS' not in sentence:
                                            sim = sim - 0.01
                                        sim2id[sim] = count
                                        sims.append(sim)
                                        count += 1
                                        self.status = '2'
                                    # '''提取最相似内容 车系，年款，排量，变速箱'''
                                    elif year in content and fuel_consumption in content \
                                            and fuel_consumption != '' and biansu in senti_words_biansu_content:
                                        # and bangxing in content and self.pailiang_judge(pailiang_list, content_pailiang_list) is True
                                        id2content[count] = content
                                        cn = self.seg.cut(
                                            content.upper().replace(' ', '').replace('\r', '').replace('\u3000',
                                                                                                       ''))
                                        # cn = list(jieba.cut(
                                        # content.upper().replace('\r', '').replace(' ', '').replace('\u3000', '')))
                                        sim = (self.simer.distance(word_list, cn))
                                        if 'PLUS' in content and 'PLUS' in sentence:
                                            sim += 0.01
                                        elif 'PLUS' in content and 'PLUS' not in sentence:
                                            sim = sim - 0.01
                                        # print(sim)
                                        sim2id[sim] = count
                                        sims.append(sim)
                                        count += 1
                                        self.status = '3'
                                    elif '电动' in sentence and year in content:
                                        # and bangxing in content and lwh == lwh_content
                                        id2content[count] = content
                                        cn = self.seg.cut(
                                            content.upper().replace(' ', '').replace('\r', '').replace('\u3000',
                                                                                                       ''))
                                        # cn = list(jieba.cut(
                                        # content.upper().replace('\r', '').replace(' ', '').replace('\u3000', '')))
                                        sim = (self.simer.distance(word_list, cn))
                                        if 'PLUS' in content and 'PLUS' in sentence:
                                            sim += 0.01
                                        elif 'PLUS' in content and 'PLUS' not in sentence:
                                            sim = sim - 0.01
                                        sim2id[sim] = count
                                        sims.append(sim)
                                        count += 1
                                        self.status = '3'
                                    else:
                                        continue
                                        self.status = 'None'

                        except:
                            pass
                    else:
                        continue
            else:
                id2content = {}
                sim2id = {}
                chexi_list=[i.strip() for i in open(self.ChexiPath,encoding='utf-8') if i.strip()]
                for value in chexi_list:
                    if value in sentence:
                        value = '{}'.format(value)
                        commandText = {'FIRST_CARS_QUKONG': {'$regex': value}}
                        cs_cb_data = self.export_data_list(commandText)
                        if len(cs_cb_data):
                            try:
                                year = str(re.search('.*(19\d{2}|20\d{2})[款 ]', sentence).group(1))
                                try:
                                    fuel_consumption = str(re.search('.*?(\d{1}.\d{1})[TL]', sentence).group(1))
                                except:
                                    fuel_consumption = ''
                                # seats = str(re.search('(\d{1}座)', sentence))
                                # if senti_words_banxing==[]:
                                #     senti_words_banxing=['']
                                for p in range(len(senti_words_banxing)):
                                    bangxing = senti_words_banxing[p]
                                    count = 0
                                    for j in range(len(cs_cb_data)):
                                        # cs_cb_data['STD_MODEL_TOGETHER_ADD']
                                        # content = str(cs_cb_data['STD_MODEL_TOGETHER_ADD'][j])
                                        content = str(cs_cb_data[j])
                                        content = content.upper()
                                        lwh_content = str(re.search('(\d{0,4}\*\d{0,4}\*\d{0,4})', content).group(1))
                                        # banxing_content, content_banixng_list = self.banixng_standard(content)
                                        pailiang_content, content_pailiang_list = self.pailiang_standard(content)
                                        # if senti_words_banxing!=[]:
                                        # for banxing in senti_words_banxing:
                                        # if banxing in content:
                                        flag_biansu, word_list_biansu, senti_words_biansu = self.check_biansu(sentence)
                                        flag_biansu_content, word_list_biansu_content, senti_words_biansu_content = self.check_biansu(
                                            content)
                                        try:
                                            if flag_biansu:
                                                biansu = senti_words_biansu[0]
                                        except:
                                            biansu = ''
                                        if year in content and fuel_consumption in content \
                                                and fuel_consumption != '' and lwh == lwh_content and biansu in senti_words_biansu_content \
                                                and bangxing in content:
                                            # and bangxing in content and self.pailiang_judge(pailiang_list, content_pailiang_list) is True
                                            id2content[count] = content
                                            cn = self.seg.cut(
                                                content.upper().replace(' ', '').replace('\r', '').replace('\u3000', ''))
                                            # cn = list(jieba.cut(
                                            # content.upper().replace('\r', '').replace(' ', '').replace('\u3000', '')))
                                            sim = (self.simer.distance(word_list, cn))
                                            # print(sim)
                                            if 'PLUS' in content and 'PLUS' in sentence:
                                                sim = sim + 0.01
                                            elif 'PLUS' in content and 'PLUS' not in sentence:
                                                sim = sim - 0.01
                                            sim2id[sim] = count
                                            sims.append(sim)
                                            count += 1
                                            self.status = '1'
                                        elif '电动' in sentence and year in content \
                                                and lwh == lwh_content and bangxing in content:
                                            # and bangxing in content and lwh == lwh_content
                                            id2content[count] = content
                                            cn = self.seg.cut(
                                                content.upper().replace(' ', '').replace('\r', '').replace('\u3000', ''))
                                            # cn = list(jieba.cut(
                                            # content.upper().replace('\r', '').replace(' ', '').replace('\u3000', '')))
                                            sim = (self.simer.distance(word_list, cn))
                                            if 'PLUS' in content and 'PLUS' in sentence:
                                                sim = sim + 0.01
                                            elif 'PLUS' in content and 'PLUS' not in sentence:
                                                sim = sim - 0.01
                                            sim2id[sim] = count
                                            sims.append(sim)
                                            count += 1
                                            self.status = '1'
                                        #   '''提取最相似内容 车系，年款，排量，长宽高，变速箱'''
                                        elif year in content and fuel_consumption in content \
                                                and fuel_consumption != '' and lwh == lwh_content and biansu in senti_words_biansu_content:
                                            # and bangxing in content and self.pailiang_judge(pailiang_list, content_pailiang_list) is True
                                            id2content[count] = content
                                            cn = self.seg.cut(
                                                content.upper().replace(' ', '').replace('\r', '').replace('\u3000',
                                                                                                           ''))
                                            # cn = list(jieba.cut(
                                            # content.upper().replace('\r', '').replace(' ', '').replace('\u3000', '')))
                                            sim = (self.simer.distance(word_list, cn))
                                            if 'PLUS' in content and 'PLUS' in sentence:
                                                sim += 0.01
                                            elif 'PLUS' in content and 'PLUS' not in sentence:
                                                sim = sim - 0.01
                                            # print(sim)
                                            sim2id[sim] = count
                                            sims.append(sim)
                                            count += 1
                                            self.status = '2'
                                        elif '电动' in sentence and year in content \
                                                and lwh == lwh_content:
                                            # and bangxing in content and lwh == lwh_content
                                            id2content[count] = content
                                            cn = self.seg.cut(
                                                content.upper().replace(' ', '').replace('\r', '').replace('\u3000',
                                                                                                           ''))
                                            # cn = list(jieba.cut(
                                            # content.upper().replace('\r', '').replace(' ', '').replace('\u3000', '')))
                                            sim = (self.simer.distance(word_list, cn))
                                            if 'PLUS' in content and 'PLUS' in sentence:
                                                sim += 0.01
                                            elif 'PLUS' in content and 'PLUS' not in sentence:
                                                sim = sim - 0.01
                                            sim2id[sim] = count
                                            sims.append(sim)
                                            count += 1
                                            self.status = '2'
                                        # '''提取最相似内容 车系，年款，排量，变速箱'''
                                        elif year in content and fuel_consumption in content \
                                                and fuel_consumption != '' and biansu in senti_words_biansu_content:
                                            # and bangxing in content and self.pailiang_judge(pailiang_list, content_pailiang_list) is True
                                            id2content[count] = content
                                            cn = self.seg.cut(
                                                content.upper().replace(' ', '').replace('\r', '').replace('\u3000',
                                                                                                           ''))
                                            # cn = list(jieba.cut(
                                            # content.upper().replace('\r', '').replace(' ', '').replace('\u3000', '')))
                                            sim = (self.simer.distance(word_list, cn))
                                            if 'PLUS' in content and 'PLUS' in sentence:
                                                sim += 0.01
                                            elif 'PLUS' in content and 'PLUS' not in sentence:
                                                sim = sim - 0.01
                                            # print(sim)
                                            sim2id[sim] = count
                                            sims.append(sim)
                                            count += 1
                                            self.status = '3'
                                        elif '电动' in sentence and year in content:
                                            # and bangxing in content and lwh == lwh_content
                                            id2content[count] = content
                                            cn = self.seg.cut(
                                                content.upper().replace(' ', '').replace('\r', '').replace('\u3000',
                                                                                                           ''))
                                            # cn = list(jieba.cut(
                                            # content.upper().replace('\r', '').replace(' ', '').replace('\u3000', '')))
                                            sim = (self.simer.distance(word_list, cn))
                                            if 'PLUS' in content and 'PLUS' in sentence:
                                                sim += 0.01
                                            elif 'PLUS' in content and 'PLUS' not in sentence:
                                                sim = sim - 0.01
                                            sim2id[sim] = count
                                            sims.append(sim)
                                            count += 1
                                            self.status='3'
                                        else:
                                            continue
                                            self.status = 'None'
                            except:
                                pass
                        else:
                            continue
        try:
            #sims=[round(r,2) for r in sims]
            max_sim = max(sims)
            sim_w = round(max_sim, 2)
            if sim_w > 0.60:
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
    handle=Doc_most_sim()

    with open('data/jingyou_cxpp_outputa1.csv','w',encoding='utf-8') as writer:
        results=[]
        for i,line in enumerate(open('data/jingyou_cxpp_wpp.csv',encoding='utf-8')):
            content =line.strip()
            #print(content)
            start_time = datetime.datetime.now()
            result=handle.doc_score_all(content)
            writer.write(content + 'CUTLINE' + result+ '\n')
            # results.append(str(result))
            end_time = datetime.datetime.now()
            s=str(i)+'\t'+str((end_time - start_time).seconds)
            print(s)


#        writer.write('\n'.join(results))

    # for line in ['北京现代 现代悦动 2008款 1.6L MT 舒适型4542*1775*1490国IV',
    #              '北京现代 现代悦动 2008款 1.6L AT 舒适型4542*1775*1490国IV',
    #              '北京现代 现代悦动 2008款 1.6L MT 豪华型4542*1775*1490国IV',
    #              '北京现代 现代悦动 2008款 1.6L AT 舒适型天窗版4542*1775*1490国IV',
    #              '北京现代 现代悦动 2008款 1.6L MT 豪华型真皮版4542*1775*1490国IV',
    #              '北京现代 现代悦动 2008款 1.6L MT 豪华型导航版4542*1775*1490国IV',
    #              '北京现代 现代悦动 2008款 1.6L MT 豪华型真皮导航版4542*1775*1490国IV',
    #              '北京现代 现代悦动 2008款 1.6L AT 豪华型4542*1775*1490国IV',
    #              '北京现代 现代悦动 2008款 1.6L AT 豪华型导航版4542*1775*1490国IV']:
    # for line in [ '名爵 名爵名爵6新能源 2019款 掀背 50T 1.5T 自动 PLUS前置前驱 4695*1848*1458 AT 5门5座1.5T国Ⅵ']:
    #     content =line.strip()
    #     start_time = datetime.datetime.now()
    #     content=content.upper()
    #     result=handle.doc_score_all(content)
    #     end_time = datetime.datetime.now()
    #     s = str((end_time - start_time).seconds)
    #     print(s)
    #     print(result)