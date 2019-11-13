#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from bert_serving.client import BertClient
bc = BertClient()
sentence_vector=bc.encode(['北京现代现代悦动2008款1.6LAT舒适型天窗版4542*1775*1490国IV'])
print(sentence_vector)