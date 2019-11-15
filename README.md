# KBQA-BERT
## 基于知识图谱的问答系统，BERT做命名实体识别和句子相似度，分为online和outline模式

## Introduction

**把过去一段时间同学们遇到的主要问题汇总一下，下面是一些FAQ：**  
  
**Q:** 运行run_ner.py时未找到dev.txt,请问这个文件是怎么生成的呢？  
**A:** 这一部分我记得当初是没有足够多的数据，我把生成的test.txt copy, 改成dev.txt了。  
  
**Q:** 你好，我下载了你的项目，但在运行run_ner的时候总是会卡在Saving checkpoint 0 to....这里，请问是什么原因呢？  
**A:** ner部分是存在一些问题，我也没有解决，但是我没有遇到这种情况。微调bert大概需要12GB左右的显存，大家可以把batch_size和max_length调小一点，说不定会解决这个问题！。  
  
**Q:** 该项目有没有相应的论文呢？  
**A:** 回答是肯定的，有的，送上 [**论文传送门**!](http://www.cnki.com.cn/Article/CJFDTotal-DLXZ201705041.htm)  

**Q:** 数据下载失败，不满足现有数据？  
**A:** 数据在Data中，更多的数据在[**NLPCC2016**](http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html) 和[**NLPCC2017**](http://tcci.ccf.org.cn/conference/2017/taskdata.php)。    

### 环境配置

    Python版本为3.6
    tensorflow版本为1.13
    XAMPP版本为3.3.2
    Navicat Premium12
    
### 目录说明

    bert文件夹是google官方下载的
    Data文件夹存放原始数据和处理好的数据
        construct_dataset.py  生成NER_Data的数据
        construct_dataset_attribute.py  生成Sim_Data的数据
        triple_clean.py  生成三元组数据
        load_dbdata.py  将数据导入mysql db
    ModelParams文件夹需要下载BERT的中文配置文件：chinese_L-12_H-768_A-12
    Output文件夹存放输出的数据
    
    基于BERT的命名实体识别模块
    - lstm_crf_layer.py
    - run_ner.py
    - tf_metrics.py
    - conlleval.py
    - conlleval.pl
    - run_ner.sh
    
    基于BERT的句子相似度计算模块
    - args.py
    - run_similarity.py
    
    KBQA模块
    - terminal_predict.py
    - terminal_ner.sh
    - kbqa_test.py
	
    ***基于BERT的车型相似度计算
	-bert-serving安装 参考网址 https://www.jianshu.com/p/973822fea15f
	-打开一个终端  bert-serving-start -model_dir D:\code\DADI-BERT-NER\ModelParams\chinese_L-12_H-768_A-12/ -num_worker=1
	-construct_dataset.py  生成NER_Data的数据
	ModelParams文件夹需要下载BERT的中文配置文件：chinese_L-12_H-768_A-12
    Output文件夹存放输出的数据
	- python run_ner.py \
      --task_name=ner \
      --data_dir=./Data/CHEXI_NER_Data \
      --vocab_file=./ModelParams/chinese_L-12_H-768_A-12/vocab.txt \
      --bert_config_file=./ModelParams/chinese_L-12_H-768_A-12/bert_config.json \
      --output_dir=./Output/NER \
      --init_checkpoint=./ModelParams/chinese_L-12_H-768_A-12/bert_model.ckpt \
      --data_config_path=./Config/NER/ner_data.conf \
      --do_train=True \
      --do_eval=False \
      --max_seq_length=88 \
      --lstm_size=128 \
      --num_layers=1 \
      --train_batch_size=32 \
      --eval_batch_size=8 \
      --predict_batch_size=8 \
      --learning_rate=5e-5 \
      --num_train_epochs=1 \
      --droupout_rate=0.5 \
      --clip=5
      车型相似度匹配
      数据库为MongoDB
    - similarity_calculation_jingyou.py \
      --task_name=ner \
      --data_dir=./Data/CHEXI_NER_Data \
      --vocab_file=./ModelParams/chinese_L-12_H-768_A-12/vocab.txt \
      --bert_config_file=./ModelParams/chinese_L-12_H-768_A-12/bert_config.json \
      --output_dir=./Output/NER \
      --init_checkpoint=./ModelParams/chinese_L-12_H-768_A-12/bert_model.ckpt \
      --data_config_path=./Config/NER/ner_data.conf \
      --do_train=False \
      --do_eval=False \
      --max_seq_length=88 \
      --lstm_size=128 \
      --num_layers=1 \
      --train_batch_size=32 \
      --eval_batch_size=8 \
      --predict_batch_size=8 \
      --learning_rate=5e-5 \
      --num_train_epochs=1 \
      --droupout_rate=0.5 \
      --clip=5
    
 ### 使用说明
    
    - run_ner.sh
    NER训练和调参
    
    - terminal_ner.sh
    do_predict_online=True  NER线上预测
    do_predict_outline=True  NER线下预测
    
    - args.py
    train = True  预训练模型
    test = True  SIM线上测试
    
    - run_similarity.py
    python run一下就可以啦
    
    - kbqa_test.py
    基于KB的问答测试
  

    
