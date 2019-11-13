# coding:utf-8
import sys
import os
import pandas as pd


'''
构造NER训练集，实体序列标注，训练BERT+BiLSTM+CRF
'''
# [training, testing]
data_type = "training"
file = "./CHEXI_NER_Data/销售车型.xlsx"

q_t_a_list = []
seq_list = []    #["中","华","人","民"]
seq_tag_list = []  #[0,0,1,1]

datas=pd.read_excel(file,encoding='utf-8').reset_index(drop=True)
negtive_list=[]
for row in datas.index:
    line,chexi,banxing=datas.loc[row][["车型描述","车系","版型"]]
    line=str(line).replace('\u3000', '').replace('\xc2\xa0', '').replace(' ', '').lower()
    chexi=str(chexi).lower()
    banxing=str(banxing).lower()
    if chexi in line:
        entities = chexi
        content=line
        chexi_list = list(content)
        seq_list.extend(chexi_list)
        tag_list = ["O" for i in range(len(chexi_list))]
        tag_start_index =content.find(entities)
        for i in range(tag_start_index, tag_start_index+len(entities)):
            if tag_start_index == i:
                tag_list[i] = "B-LOC"
            else:
                tag_list[i] = "I-LOC"
        banxings=banxing.split(' ')
        for bx in banxings:
            if bx in line:
                entities = bx
                tag_start_index = content.find(entities)
                for i in range(tag_start_index, tag_start_index + len(entities)):
                    if tag_start_index == i:
                        tag_list[i] = "B-ORG"
                    else:
                        tag_list[i] = "I-ORG"
    else:
        negtive_list.append([line,chexi,banxing])
        continue
    seq_tag_list.extend(tag_list)
    seq_list.extend([" "])
    seq_tag_list.extend([" "])
    q_t_a_list.append([line,chexi,banxing])

print('\t'.join(seq_tag_list[0:50]))
print('\t'.join(seq_list[0:50]))
seq_result = [str(q)+" "+tag for q, tag in zip(seq_list, seq_tag_list)]
data_type=data_type.replace('ing','')
with open("./CHEXI_NER_Data/"+data_type+".txt", "w", encoding='utf-8') as f:
    f.write("\n".join(seq_result))


df = pd.DataFrame(q_t_a_list, columns=["MODEL_STANDARD_NAME","CATE_NAME","VERSION"])
df.to_csv("./CHEXI_NER_Data/df_"+data_type+".csv", encoding='utf-8', index=False)

df1=pd.DataFrame(negtive_list, columns=["MODEL_STANDARD_NAME","CATE_NAME","VERSION"])
df1.to_csv("./CHEXI_NER_Data/df_negtive.csv", encoding='utf-8', index=False)