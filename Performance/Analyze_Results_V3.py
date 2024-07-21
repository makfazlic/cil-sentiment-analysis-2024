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

Num_Of_Iterations=100000 #int(input("How many iterations: "))



Data_Label0=[]
Data_Label1=[]
url_struct_found=[0,0]

with open("./train_neg_full.txt", 'r',encoding='utf-8') as infile:
    text=infile.read().split("\n")[:-1]
    #print(len(text))
    for i in text:
        if "<url>" in i and (i.count("(")>i.count(")")):
            url_struct_found[0]+=1
        else:
            Data_Label0.append(i) #Remove_User_At_Beginning(i))


with open("./train_pos_full.txt", 'r',encoding='utf-8') as infile:
    text=infile.read().split("\n")[:-1]
    #print(len(text))
    for i in text:
        if "<url>" in i and (i.count("(")>i.count(")")):
            url_struct_found[1]+=1
        else:
            Data_Label1.append(i) #Remove_User_At_Beginning(i))
#print(url_struct_found)
#exit()
#Duplicate removal
Data_Label0=list(set(Data_Label0))
Data_Label1=list(set(Data_Label1))

#Test Train Split
Train_Data_Text_L0, Test_Data_Text_L0 = train_test_split(Data_Label0, test_size=0.33, random_state=42)
Train_Data_Text_L1, Test_Data_Text_L1 = train_test_split(Data_Label1, test_size=0.33, random_state=42)


Data_Text=[Test_Data_Text_L0,Test_Data_Text_L1]




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

"""
Data_Text=[[],[]]
with open("./train_neg_full.txt", 'r',encoding='utf-8') as infile:
    text=infile.read().split("\n")[:-1]
    for i in text:
        if not ("<url>" in i and (i.count("(")>i.count(")"))):
            Data_Text[0].append(Remove_User_At_Beginning(i))
with open("./train_pos_full.txt", 'r',encoding='utf-8') as infile:
    text=infile.read().split("\n")[:-1]
    for i in text:
        if not ("<url>" in i and (i.count("(")>i.count(")"))):
            Data_Text[1].append(Remove_User_At_Beginning(i))
            
"""

cos_d1 = nn.CosineSimilarity(dim=1, eps=1e-6)

Results={}

Results["Full_Hierarchy"]={}
Results["Full_Hierarchy"]["Total"]=0
Results["Full_Hierarchy"]["Correct"]=0
Results["Full_Hierarchy"]["Total_0"]=0
Results["Full_Hierarchy"]["Correct_0"]=0
Results["Full_Hierarchy"]["Total_1"]=0
Results["Full_Hierarchy"]["Correct_1"]=0
Results["Full_Hierarchy"]["Ensembling_Correct"]=0
Results["Full_Hierarchy"]["Ensembling_Correct_0"]=0
Results["Full_Hierarchy"]["Ensembling_Correct_1"]=0


for i in range(len(config_data)):
    Results["Level_"+str(i)]={}
    Results["Level_"+str(i)]["Total_Seen"]=0
    Results["Level_"+str(i)]["Total_Kept"]=0
    Results["Level_"+str(i)]["Correct_Seen"]=0
    Results["Level_"+str(i)]["Correct_Kept"]=0
    Results["Level_"+str(i)]["Total_Seen_0"]=0
    Results["Level_"+str(i)]["Total_Kept_0"]=0
    Results["Level_"+str(i)]["Correct_Seen_0"]=0
    Results["Level_"+str(i)]["Correct_Kept_0"]=0
    Results["Level_"+str(i)]["Total_Seen_1"]=0
    Results["Level_"+str(i)]["Total_Kept_1"]=0
    Results["Level_"+str(i)]["Correct_Seen_1"]=0
    Results["Level_"+str(i)]["Correct_Kept_1"]=0
    Results["Level_"+str(i)]["Clusters"]={}

ac_iteration=0
while ac_iteration<Num_Of_Iterations:
    print(round(100*ac_iteration/Num_Of_Iterations),end="\r")
    #print("started")
    rl = random.randint(0,1)
    rt = random.randint(0,len(Data_Text[rl])-1)
    ac_Averaging=[]
    for ac_level_number,ac_level in enumerate(config_data):
        
        try:
            with torch.no_grad():
                Tokenized=ac_level[3].tokenize([Data_Text[rl][rt]])
                Embeddings=ac_level[3](Tokenized)["sentence_embedding"]
                Prediction_nR=ac_level[4](Embeddings).item()
                Prediction=round(Prediction_nR)
                Embeddings=Embeddings[0]
        except:
            break
        #print("hi")
        vecs1=Embeddings.unsqueeze(0).repeat(ac_level[0].shape[0],1)
        vecs2=ac_level[0]
        cosSimis=cos_d1(vecs1, vecs2) 
        maxSimi=torch.max(cosSimis,dim=0)
        acCluster=maxSimi.indices.item()
        
        ac_Averaging.append(Prediction_nR)
        
        Cluster_Key="Cluster_"+str(acCluster)
        if Cluster_Key not in Results["Level_"+str(ac_level_number)]["Clusters"]:
            Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]={}
            Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Biased"]=(acCluster in ac_level[1])
            Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Total"]=0
            Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Correct"]=0
            Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Total_0"]=0
            Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Correct_0"]=0
            Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Total_1"]=0
            Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Correct_1"]=0
            Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Sentences"]=[]
        Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Sentences"].append(Data_Text[rl][rt])
        Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Total"]+=1
        Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Correct"]+=1-abs(rl-Prediction)
        if rl==0:
            Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Total_0"]+=1
            Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Correct_0"]+=1-abs(rl-Prediction)
        else:
            Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Total_1"]+=1
            Results["Level_"+str(ac_level_number)]["Clusters"][Cluster_Key]["Correct_1"]+=1-abs(rl-Prediction)
        
        Results["Level_"+str(ac_level_number)]["Total_Seen"]+=1
        Results["Level_"+str(ac_level_number)]["Correct_Seen"]+=1-abs(rl-Prediction)
        if rl==0:
            Results["Level_"+str(ac_level_number)]["Total_Seen_0"]+=1
            Results["Level_"+str(ac_level_number)]["Correct_Seen_0"]+=1-abs(rl-Prediction)
        else:
            Results["Level_"+str(ac_level_number)]["Total_Seen_1"]+=1
            Results["Level_"+str(ac_level_number)]["Correct_Seen_1"]+=1-abs(rl-Prediction)
        
        if acCluster in ac_level[1] or ac_level_number>=len(config_data)-1:
            Results["Full_Hierarchy"]["Total"]+=1
            Results["Full_Hierarchy"]["Correct"]+=1-abs(rl-Prediction)
            averaged_prediction=round(sum(ac_Averaging)/len(ac_Averaging))
            Results["Full_Hierarchy"]["Ensembling_Correct"]+=1-abs(rl-averaged_prediction)
            if rl==0:
                Results["Full_Hierarchy"]["Total_0"]+=1
                Results["Full_Hierarchy"]["Correct_0"]+=1-abs(rl-Prediction)
                Results["Full_Hierarchy"]["Ensembling_Correct_0"]+=1-abs(rl-averaged_prediction)
            else:
                Results["Full_Hierarchy"]["Total_1"]+=1
                Results["Full_Hierarchy"]["Correct_1"]+=1-abs(rl-Prediction)
                Results["Full_Hierarchy"]["Ensembling_Correct_1"]+=1-abs(rl-averaged_prediction)

            Results["Level_"+str(ac_level_number)]["Total_Kept"]+=1
            Results["Level_"+str(ac_level_number)]["Correct_Kept"]+=1-abs(rl-Prediction)
            if rl==0:
                Results["Level_"+str(ac_level_number)]["Total_Kept_0"]+=1
                Results["Level_"+str(ac_level_number)]["Correct_Kept_0"]+=1-abs(rl-Prediction)
            else:
                Results["Level_"+str(ac_level_number)]["Total_Kept_1"]+=1
                Results["Level_"+str(ac_level_number)]["Correct_Kept_1"]+=1-abs(rl-Prediction)
            ac_iteration+=1
            #print("hey")
            break

if Results["Full_Hierarchy"]["Total"]!=0:
    Results["Full_Hierarchy"]["Accuracy"]=Results["Full_Hierarchy"]["Correct"]/Results["Full_Hierarchy"]["Total"]
    Results["Full_Hierarchy"]["Accuracy_Ensembling"]=Results["Full_Hierarchy"]["Ensembling_Correct"]/Results["Full_Hierarchy"]["Total"]
if Results["Full_Hierarchy"]["Total_0"]!=0:
    Results["Full_Hierarchy"]["Accuracy_0"]=Results["Full_Hierarchy"]["Correct_0"]/Results["Full_Hierarchy"]["Total_0"]
    Results["Full_Hierarchy"]["Accuracy_Ensembling_0"]=Results["Full_Hierarchy"]["Ensembling_Correct_0"]/Results["Full_Hierarchy"]["Total_0"]
if Results["Full_Hierarchy"]["Total_1"]!=0:
    Results["Full_Hierarchy"]["Accuracy_1"]=Results["Full_Hierarchy"]["Correct_1"]/Results["Full_Hierarchy"]["Total_1"]
    Results["Full_Hierarchy"]["Accuracy_Ensembling_1"]=Results["Full_Hierarchy"]["Ensembling_Correct_1"]/Results["Full_Hierarchy"]["Total_1"]

for i in Results:
    if i=="Full_Hierarchy":
        continue
    if Results[i]["Total_Seen"]!=0:
        Results[i]["Accuracy_Seen"]=Results[i]["Correct_Seen"]/Results[i]["Total_Seen"]
    if Results[i]["Total_Seen_0"]!=0:
        Results[i]["Accuracy_Seen_0"]=Results[i]["Correct_Seen_0"]/Results[i]["Total_Seen_0"]
    if Results[i]["Total_Seen_1"]!=0:
        Results[i]["Accuracy_Seen_1"]=Results[i]["Correct_Seen_1"]/Results[i]["Total_Seen_1"]
    if Results[i]["Total_Kept"]!=0:
        Results[i]["Accuracy_Kept"]=Results[i]["Correct_Kept"]/Results[i]["Total_Kept"]
    if Results[i]["Total_Kept_0"]!=0:
        Results[i]["Accuracy_Kept_0"]=Results[i]["Correct_Kept_0"]/Results[i]["Total_Kept_0"]
    if Results[i]["Total_Kept_1"]!=0:
        Results[i]["Accuracy_Kept_1"]=Results[i]["Correct_Kept_1"]/Results[i]["Total_Kept_1"]

    for j in Results[i]["Clusters"]:
        if Results[i]["Clusters"][j]["Total"]!=0:
            Results[i]["Clusters"][j]["Accuracy"]=Results[i]["Clusters"][j]["Correct"]/Results[i]["Clusters"][j]["Total"]
        if Results[i]["Clusters"][j]["Total_0"]!=0:
            Results[i]["Clusters"][j]["Accuracy_0"]=Results[i]["Clusters"][j]["Correct_0"]/Results[i]["Clusters"][j]["Total_0"]
        if Results[i]["Clusters"][j]["Total_1"]!=0:
            Results[i]["Clusters"][j]["Accuracy_1"]=Results[i]["Clusters"][j]["Correct_1"]/Results[i]["Clusters"][j]["Total_1"]

with open("Determine_Clustering_Results.json","w") as f:
    json.dump(Results,f,sort_keys=False,indent=1)
print(json.dumps(Results,sort_keys=False,indent=1),flush=True)
