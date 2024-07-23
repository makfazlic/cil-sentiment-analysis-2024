import torch
from torch import nn
import time
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, TaskType, get_peft_model
import adapters
from adapters import PrefixTuningConfig
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import random
from sklearn.cluster import KMeans
import json
import sys
import math
import copy
import numpy as np
from numpy import nan
import csv

with open("./test_data.txt", 'r',encoding='utf-8') as infile:
    text=infile.read().split("\n")[:-1]

for i in range(len(text)):
    if text[i].count(",")==0:
        print("Error")
        exit()
    pos_K=text[i].find(",")+1
    text[i]=text[i][pos_K:]

print(len(text))
print(text[0])
#exit()
print()
with open('sample_submission.csv', 'r') as f:
    predictions = list(csv.reader(f, delimiter=","))
print(len(predictions))
print(predictions[0])
print()

#Helper classes and functions:
class Adaptor_Net(nn.Module):
    def __init__(self,Genom,Input_Size):
        super(Adaptor_Net, self).__init__()
        Layer_List=[]
        for i in range(Genom[2]):
            Layer_In_Size=None
            if i==0:
                Layer_In_Size=Input_Size
            elif i==1:
                Layer_In_Size=Genom[3]
            elif i==2:
                Layer_In_Size=Genom[4]

            Layer_Out_Size=None
            if i==Genom[2]-1:
                Layer_Out_Size=1
            elif i==0:
                Layer_Out_Size=Genom[3]
            elif i==1:
                Layer_Out_Size=Genom[4]
            Layer_List.append(nn.Linear(Layer_In_Size, Layer_Out_Size))
        self.Layer_List=nn.ModuleList(Layer_List)

        self.Activation_Function = None
        if Genom[7]=="ReLU":
            self.Activation_Function=nn.ReLU()
        elif Genom[7]=="Sigmoid":
            self.Activation_Function=nn.Sigmoid()
        elif Genom[7]=="Tanh":
            self.Activation_Function=nn.Tanh()

        self.sig=nn.Sigmoid()

    # x represents our data
    def forward(self, x):
        for Ac_Layer_Pos in range(len(self.Layer_List)):
            x=self.Layer_List[Ac_Layer_Pos](x)
            if Ac_Layer_Pos!=len(self.Layer_List)-1:
                x=self.Activation_Function(x)
        return self.sig(x)






with open("local_models/own_models/Configuration_Biasing_BS_2_TS_12000.json","r") as f:
    config_data=json.load(f)


for i in range(len(config_data)):
    config_data[i][0]=torch.as_tensor(config_data[i][0])
    config_data[i].append(SentenceTransformer(config_data[i][2][0]))
    config_data[i].append(torch.load(config_data[i][2][1]))
    #print(len(config_data[i]))


user_at_beginning=0
def Remove_User_At_Beginning(ac_string):
    global user_at_beginning
    while True:
        if len(ac_string)==0:
            break
        elif ac_string[0]==" " or ac_string[0]==",":
            ac_string=ac_string[1:]
        else:
            if len(ac_string)<2:
               break
            elif ac_string[:2]=="rt":
                 ac_string=ac_string[2:]
            elif len(ac_string)<6:
                break
            elif ac_string[:6]=="<user>":
                ac_string=ac_string[6:]
                user_at_beginning+=1
            else:
                break
    return ac_string


cos_d1 = nn.CosineSimilarity(dim=1, eps=1e-6)

prediction_share=[0,0,0,0,0]

for ac_text_pos,ac_text in enumerate(tqdm(text)):
    ac_text_pos=ac_text_pos+1
    if "<url>" in ac_text and (ac_text.count("(")>ac_text.count(")")):
        prediction_share[0]+=1
        predictions[ac_text_pos][1]=-1
        continue
    for ac_level_number,ac_level in enumerate(config_data):
        try:
            with torch.no_grad():
                Tokenized=ac_level[3].tokenize([ac_text])
                Embeddings=ac_level[3](Tokenized)["sentence_embedding"]
                Prediction_nR=ac_level[4](Embeddings).item()
                Prediction=round(Prediction_nR)
                Embeddings=Embeddings[0]
        except:
            prediction_share[1]+=1
            predictions[ac_text_pos][1]=-1
            break
        
        vecs1=Embeddings.unsqueeze(0).repeat(ac_level[0].shape[0],1)
        vecs2=ac_level[0]
        cosSimis=cos_d1(vecs1, vecs2) 
        maxSimi=torch.max(cosSimis,dim=0)
        acCluster=maxSimi.indices.item()
        
        if acCluster in ac_level[1] or ac_level_number>=len(config_data)-1:
            prediction_share[2+ac_level_number]+=1
            if Prediction==0:
                predictions[ac_text_pos][1]=-1
            elif Prediction==1:
                predictions[ac_text_pos][1]=1
            else:
                print("Error")
                exit()
            
            break

with open('my_submission.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(predictions)


print(prediction_share,flush=True)
