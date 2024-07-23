#!/usr/bin/env python
# coding: utf-8

#Todo:
#Exchange
#Max_Position=min(len(Test_Samples[0]),len(Test_Samples[1]))-1
#        while max(Actual_Position_Test)<=Max_Position:
#with
#while Actual_Position_Test[0]<=Test_Samples[0] and Actual_Position_Test[1]<=Test_Samples[1]

#No grave mistake just improves code

###################
#    Libraries    #
###################

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



###################################
#    Train and Eval Parameters    #
###################################

Time_To_Evalu               = 60*15                 # Test for 15 minuntes
Time_To_Initialize_Clusters = 60*30                 # Initialize for 30 Min
Biasing_Determining_Time    = 60*30                 # Determine Bias for 30 min
Device_of_Model             = "cpu"                 # "cuda" for gpu
Number_Training_Batches     = 6000*2
Save_Models_Times           = 10



#####################
#    Neural Nets    #
#####################

class Adaptor_Net(nn.Module):
    def __init__(self,Genom,Input_Size):
        super(Adaptor_Net, self).__init__()
        Layer_List=[]
        for i in range(Genom["Depth_Adapter"]):
            Layer_In_Size=None
            if i==0:
                Layer_In_Size=Input_Size
            elif i==1:
                Layer_In_Size=Genom["Width_First_Layer"]
            elif i==2:
                Layer_In_Size=Genom["Width_Second_Layer"]

            Layer_Out_Size=None
            if i==Genom["Depth_Adapter"]-1:
                Layer_Out_Size=1
            elif i==0:
                Layer_Out_Size=Genom["Width_First_Layer"]
            elif i==1:
                Layer_Out_Size=Genom["Width_Second_Layer"]
            Layer_List.append(nn.Linear(Layer_In_Size, Layer_Out_Size))
        self.Layer_List=nn.ModuleList(Layer_List)

        self.Activation_Function = None
        if Genom["Activation_Function_Adapter_Model"]=="ReLU":
            self.Activation_Function=nn.ReLU()
        elif Genom["Activation_Function_Adapter_Model"]=="Sigmoid":
            self.Activation_Function=nn.Sigmoid()
        elif Genom["Activation_Function_Adapter_Model"]=="Tanh":
            self.Activation_Function=nn.Tanh()

        self.sig=nn.Sigmoid()

    # x represents our data
    def forward(self, x):
        for Ac_Layer_Pos in range(len(self.Layer_List)):
            x=self.Layer_List[Ac_Layer_Pos](x)
            if Ac_Layer_Pos!=len(self.Layer_List)-1:
                x=self.Activation_Function(x)
        return self.sig(x)


class Clustering_Net(nn.Module):
    def __init__(self,Embedding_Dimi,Initialization,Adapt_Plus_Percentage,Adapt_Minus_Percentage,Num_Of_Clusters=10):
        super(Clustering_Net, self).__init__()

        self.Adapt_Plus_Percentage=Adapt_Plus_Percentage
        self.Adapt_Minus_Percentage=Adapt_Minus_Percentage
        self.Num_Of_Clusters=Num_Of_Clusters
        self.Norm_Constant=self.Num_Of_Clusters*self.Num_Of_Clusters-self.Num_Of_Clusters
        self.Cluster_Means=None
        if Initialization is None:
            self.Cluster_Means=nn.Parameter(torch.rand((Num_Of_Clusters,Embedding_Dimi),requires_grad=True))
        else:
            self.Cluster_Means=nn.Parameter(torch.tensor(Initialization,requires_grad=True))
        self.Max_Modifier=torch.zeros(Num_Of_Clusters,requires_grad=False)

        self.cos_d1 = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cos_d2 = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, x):
        vecs1=x.unsqueeze(1).repeat(1, self.Num_Of_Clusters, 1)
        vecs2=self.Cluster_Means.unsqueeze(0).repeat(vecs1.shape[0], 1, 1)
        #print(vecs1.shape)
        #print(vecs2.shape)
        cosSimis=self.cos_d2(vecs1, vecs2)
        MaxModi=self.Max_Modifier.unsqueeze(0).repeat(vecs1.shape[0],1)
        cosSimis_Adapted=torch.add(cosSimis,MaxModi)
        #print("***")
        #print(self.Max_Modifier)
        #print(cosSimis)
        #print(cosSimis_Adapted)
        maxSimi=torch.max(cosSimis_Adapted,dim=1) #.values
        maxSimi_Adapted=torch.sub(maxSimi.values,self.Max_Modifier[maxSimi.indices])
        #print(maxSimi)
        #print(maxSimi_Adapted)

        vecs3=self.Cluster_Means.unsqueeze(0).repeat(self.Num_Of_Clusters,1,1)
        vecs4=self.Cluster_Means.unsqueeze(1).repeat(1,self.Num_Of_Clusters,1)
        cosSimis_CC=self.cos_d2(vecs3, vecs4)

        cosSimis_CC.fill_diagonal_(0.0)
        cosSimis_CC_sum=torch.sum(cosSimis_CC)
        cosSimis_CC_sum_normalized=torch.div(cosSimis_CC_sum,self.Norm_Constant)

        Max_Difference=(torch.max(cosSimis)-torch.min(cosSimis)).item()
        self.Max_Modifier=torch.add(self.Max_Modifier,self.Adapt_Plus_Percentage*Max_Difference)
        self.Max_Modifier[maxSimi.indices]-=(self.Adapt_Minus_Percentage*Max_Difference)
        self.Max_Modifier=torch.maximum(self.Max_Modifier,torch.zeros_like(self.Max_Modifier))
        self.Max_Modifier=torch.minimum(self.Max_Modifier,torch.full_like(self.Max_Modifier,2))
        #print(self.Max_Modifier)
        return maxSimi_Adapted,cosSimis_CC_sum_normalized




########################
#    Batch Creation    #
########################

def MakeBatches(Data,Batchsize):
    PosList_L0=list(range(len(Data[0])))
    PosList_L1=list(range(len(Data[1])))
    random.shuffle(PosList_L0)
    random.shuffle(PosList_L1)    
    return (PosList_L0,PosList_L1)

def Datapoint_Biased(ac_Datapoint,Biasing_Step_Data):
    with torch.no_grad():
        Is_Biased=False
        for Biasing_Step_DataPoint in Biasing_Step_Data:
            Tokenized=Biasing_Step_DataPoint[0].tokenize([ac_Datapoint])  
            Tokenized["input_ids"].to(Device_of_Model)
            Tokenized["attention_mask"].to(Device_of_Model)
            Embeddings=Biasing_Step_DataPoint[0](Tokenized)["sentence_embedding"].to("cpu")[0] 
            vecs1=Embeddings.unsqueeze(0).repeat(Biasing_Step_DataPoint[1].shape[0],1)
            vecs2=Biasing_Step_DataPoint[1]
            cosSimis=cos_d1(vecs1, vecs2)
            maxSimi=torch.max(cosSimis,dim=0)
            acCluster=maxSimi.indices.item()
            if acCluster in Biasing_Step_DataPoint[2]:
                Is_Biased=True
                break
    #print(Is_Biased)
    return Is_Biased        
    

def GetBatches(Actual_Position,Training_Data,Training_Samples,Seen_Datapoints,Take_Seen_One_Prob,Biasing_Step_Data,Batchsize):
    
    if Batchsize%2!=0:
        sys.exit("Batch has to be dividable by 2")

    Batchsize_h=round(Batchsize/2)
    Batch=[[],[]]

    if random.random()<Take_Seen_One_Prob and len(Seen_Datapoints)>10:
        for _ in range(Batchsize):
            Len_Seen_Datapoints=len(Seen_Datapoints)
            Position_Inserting_Datapoint=Len_Seen_Datapoints-1
            for _ in range(max(1,int(Len_Seen_Datapoints/5))):
                Position_Inserting_Datapoint=min(Position_Inserting_Datapoint,random.randint(0,Len_Seen_Datapoints-1))
            Batch[0].append(Seen_Datapoints[Position_Inserting_Datapoint][1])
            Batch[1].append(Seen_Datapoints[Position_Inserting_Datapoint][2])
            del Seen_Datapoints[Position_Inserting_Datapoint]
    else:
        neg_samples=0
        while neg_samples<Batchsize_h and Actual_Position[0]<len(Training_Samples[0]):
            if not Datapoint_Biased(Training_Data[0][Training_Samples[0][Actual_Position[0]]],Biasing_Step_Data):
                neg_samples+=1
                Batch[0].append(Training_Data[0][Training_Samples[0][Actual_Position[0]]])
                Batch[1].append(0)
            Actual_Position[0]+=1
        pos_samples=0
        while pos_samples<Batchsize_h and Actual_Position[1]<len(Training_Samples[1]):
            if not Datapoint_Biased(Training_Data[1][Training_Samples[1][Actual_Position[1]]],Biasing_Step_Data):
                pos_samples+=1
                Batch[0].append(Training_Data[1][Training_Samples[1][Actual_Position[1]]])
                Batch[1].append(1)
            Actual_Position[1]+=1
    #print("batch found")
    return Batch




########################
#    Loss Functions    #
########################

my_MSE_loss = nn.MSELoss()
def Loss_Max_Clustering(output):
    Requested_Output_ones=torch.ones_like(output)
    return my_MSE_loss(output,Requested_Output_ones)
def Loss_Cluster_Distance(output):
    Requested_Output_zeros=torch.zeros_like(output)
    output_adapted=torch.maximum(output,torch.zeros_like(output))
    return my_MSE_loss(output,Requested_Output_zeros)
def Loss_Clustering_Total(output,alpha):
    loss_1=Loss_Max_Clustering(output[0])
    loss_1=torch.mul(loss_1,alpha)

    loss_2=Loss_Cluster_Distance(output[1])
    loss_2=torch.mul(loss_2,(1-alpha))

    return torch.add(loss_1,loss_2)


cosine_simi_cal= nn.CosineSimilarity(dim=0, eps=1e-6)
def loss_embedding_similarity(Embeddings,labels):
    #print(Embeddings.shape)
    #print(labels)
    les_loss=0
    number_of_elemets_l=0
    for i_1 in range(Embeddings.shape[0]):
        for i_2 in range(i_1+1,Embeddings.shape[0]):
            if labels[i_1][0]!=labels[i_2][0]:
                les_loss=torch.add(cosine_simi_cal(Embeddings[i_1],Embeddings[i_2]),les_loss)
                #print(les_loss)
                number_of_elemets_l+=1
    les_loss=torch.div(les_loss,number_of_elemets_l)
    les_loss=torch.add(les_loss,1)
    #print(les_loss)
    #exit()
    return les_loss


#################################
#    Biasing Helper Function    #
#################################
cos_d1 = nn.CosineSimilarity(dim=1, eps=1e-6)
def Update_Biasing_Step_Data(Biasing_Step_Data,embedding_model,adapter_Model,clustering_Model,Biasing_Determining_Time,Biasing_Data,file_names):

    Cluster_Accuracy_PL={}
    Cluster_Biased=[]
    Clustering_Centers=clustering_Model.Cluster_Means.data
    with torch.no_grad():
    
        Start_Time_Biasing=time.time()
        all_datapoints_biased=0
        while time.time()-Start_Time_Biasing<Biasing_Determining_Time:
            rl=random.randint(0,1)
            rt=random.randint(0,len(Biasing_Data[rl])-1)
            if Datapoint_Biased(Biasing_Data[rl][rt],Biasing_Step_Data):
                continue
            try:
                with torch.no_grad():
                    Tokenized=embedding_model.tokenize([Biasing_Data[rl][rt]])
                    Tokenized["input_ids"].to(Device_of_Model)
                    Tokenized["attention_mask"].to(Device_of_Model)
                    Embeddings=embedding_model(Tokenized)["sentence_embedding"].to("cpu")
                    Prediction_nR=adapter_Model(Embeddings).item()
                    Prediction=round(Prediction_nR)
                    Embeddings=Embeddings[0]
            except:
                continue
            vecs1=Embeddings.unsqueeze(0).repeat(Clustering_Centers.shape[0],1)
            vecs2=Clustering_Centers
            cosSimis=cos_d1(vecs1, vecs2)
            maxSimi=torch.max(cosSimis,dim=0)
            acCluster=maxSimi.indices.item()

            if acCluster not in Cluster_Accuracy_PL:
                Cluster_Accuracy_PL[acCluster]={}
                Cluster_Accuracy_PL[acCluster]["Correct"]=0                 
                Cluster_Accuracy_PL[acCluster]["Total"]=0
            
            Cluster_Accuracy_PL[acCluster]["Total"]+=1
            all_datapoints_biased+=1
            if Prediction==1 and rl==1:
                Cluster_Accuracy_PL[acCluster]["Correct"]+=1
            elif Prediction==0 and rl==0:
                Cluster_Accuracy_PL[acCluster]["Correct"]+=1

        Cluster_Value_S10=[]
        
        for ac_key in  Cluster_Accuracy_PL:            
            if Cluster_Accuracy_PL[ac_key]["Total"]>=10 or True: #todo
                Cluster_Value_S10.append([ac_key,Cluster_Accuracy_PL[ac_key]["Correct"]/Cluster_Accuracy_PL[ac_key]["Total"]])
        Cluster_Value_S10.sort(reverse=True, key= lambda x: x[1])
        
        accounted_datapoints_biased=0
        for ac_key,_ in Cluster_Value_S10:
            if accounted_datapoints_biased<all_datapoints_biased/2:
                Cluster_Biased.append(ac_key)
                accounted_datapoints_biased+=Cluster_Accuracy_PL[ac_key]["Total"]
            else:
                break
    Biasing_Step_Data.append([])
    Biasing_Step_Data[-1].append(copy.deepcopy(embedding_model.to("cpu")).to(Device_of_Model))
    Biasing_Step_Data[-1].append(copy.deepcopy(clustering_Model.Cluster_Means.data))
    Biasing_Step_Data[-1].append(copy.deepcopy(Cluster_Biased))
    Biasing_Step_Data[-1].append(copy.deepcopy(file_names))   
    #print(Cluster_Accuracy_PL)
    #print(Cluster_Value_S10)
    #print(Biasing_Step_Data)
    #exit()
    return Biasing_Step_Data
    


########################
#    Train Function    #
########################


def Train_New_Model(Training_Data,Genom,Number_Training_Batches,Allowed_Time_Cluster_Init,Test_Data,Biasing_Step_Data,loss_statistic,Biasing_Determining_Time,Batchsize=8):

    #Statistics
    Seen_Datapoints=[]

    #Load pretrained data
    if len(Biasing_Step_Data)==0 or acGenom["Load_Finetuned_Model"]==0:
        embedding_model = SentenceTransformer(Genom["Embedding_Model"],device=Device_of_Model)
    else:
        #print("load")
        embedding_model = SentenceTransformer(Biasing_Step_Data[-1][3][0],device=Device_of_Model)


    #Extract Size of embedding
    Examples_Embedding_Size=None
    with torch.no_grad():
        Examples_Embedding_Size=embedding_model.encode(["Testing"]).shape[1]

    #Prepare Adaptor Model
    if len(Biasing_Step_Data)==0 or acGenom["Load_Finetuned_Model"]==0:
        adapter_Model=Adaptor_Net(Genom,Examples_Embedding_Size)
    else:
        #print("load2")
        adapter_Model=torch.load(Biasing_Step_Data[-1][3][1])

    #Make K-means Initialization if needed
    clustering_Model=None
    if Genom["Clustering"]==1:
        if Genom["K-Means_Initialization"]==0:
            clustering_Model=Clustering_Net(Examples_Embedding_Size,None,Genom["Increase_Cos_Similarity"],Genom["Decrease_Cos_Similarity"],Num_Of_Clusters=Genom["Cluster_Number"])
        else:
            Pos_Embeddings=[]
            Neg_Embeddings=[]
            Start_Time_Clustering_Init=time.time()
            #print("HEy")
            while Allowed_Time_Cluster_Init>time.time()-Start_Time_Clustering_Init:
                ac_sample_pos=random.randint(0, len(Training_Data[0])-1)
                if random.random()>0.5:
                    ac_sample_text=Training_Data[0][ac_sample_pos]
                    ac_sample_label=0
                else:
                    ac_sample_text=Training_Data[1][ac_sample_pos]
                    ac_sample_label=1
                if Datapoint_Biased(ac_sample_text,Biasing_Step_Data):
                    continue
                with torch.no_grad():
                    Tokenized=embedding_model.tokenize([ac_sample_text])
                    Tokenized["input_ids"].to(Device_of_Model)
                    Tokenized["attention_mask"].to(Device_of_Model)
                    Embeddings=embedding_model(Tokenized)["sentence_embedding"].to("cpu")[0].tolist()
                if ac_sample_label<0.5:
                    Neg_Embeddings.append(Embeddings)
                else:
                    Pos_Embeddings.append(Embeddings)
            Number_Of_Clusters=round(Genom["Cluster_Number"]/2)
            Clustering_init_Pos=[]
            for ac_cluster_num in range(Number_Of_Clusters, 0, -1):
                try:
                    kmeans = KMeans(n_clusters=ac_cluster_num)
                    kmeans.fit(Pos_Embeddings)
                    Clustering_init_Pos=list(kmeans.cluster_centers_)
                    break
                except:
                    print("[Error] Pos Clusering not possible",flush=True)
            Clustering_init_Neg=[]
            for ac_cluster_num in range(Number_Of_Clusters, 0, -1):
                try:
                    kmeans = KMeans(n_clusters=ac_cluster_num)
                    kmeans.fit(Neg_Embeddings)
                    Clustering_init_Neg=list(kmeans.cluster_centers_)
                    break
                except:
                    print("[Error] Neg Clustering not possible",flush=True)
            Clustering_init=Clustering_init_Pos+Clustering_init_Neg
            if len(Clustering_init)<Genom["Cluster_Number"]:
                Additional_Mean=np.random.rand(Genom["Cluster_Number"]-len(Clustering_init),Examples_Embedding_Size).tolist()
                Clustering_init+=Additional_Mean
            clustering_Model=Clustering_Net(Examples_Embedding_Size,Clustering_init,Genom["Increase_Cos_Similarity"],Genom["Decrease_Cos_Similarity"],Num_Of_Clusters=Genom["Cluster_Number"])

    
    #Prepare Learning rates and optimizer
    if Genom["Finetuning_Method"]=="None" and Genom["Clustering"]==0:
        optimizer=torch.optim.Adam([
            {'params': adapter_Model.parameters(), 'lr': Genom["Learning_Rate_Adapter_Model"]}
            ], lr=Genom["Learning_Rate_Adapter_Model"])
    elif Genom["Finetuning_Method"]=="None" and Genom["Clustering"]==1:
        optimizer=torch.optim.Adam([
            {'params': adapter_Model.parameters(), 'lr': Genom["Learning_Rate_Adapter_Model"]},
            {'params': clustering_Model.parameters(), 'lr': Genom["Learning_Rate_Clustering_Model"]}
            ], lr=Genom["Learning_Rate_Adapter_Model"])
    elif Genom["Clustering"]==0:
        optimizer=torch.optim.Adam([
            {'params': embedding_model.parameters(), 'lr': Genom["Learning_Rate_Embedding_Model"]},
            {'params': adapter_Model.parameters(), 'lr': Genom["Learning_Rate_Adapter_Model"]}
            ], lr=Genom["Learning_Rate_Embedding_Model"])
    else:
        optimizer=torch.optim.Adam([
            {'params': embedding_model.parameters(), 'lr': Genom["Learning_Rate_Embedding_Model"]},
            {'params': adapter_Model.parameters(), 'lr': Genom["Learning_Rate_Adapter_Model"]},
            {'params': clustering_Model.parameters(), 'lr': Genom["Learning_Rate_Clustering_Model"]}
            ], lr=Genom["Learning_Rate_Embedding_Model"])

    #Training loop
    criterion = nn.BCELoss()
    Training_Step_Iteration=-1
    training_finished=False
    while True:
        
        #Prepare Batches:
        Training_Samples=MakeBatches(Training_Data,Batchsize)

        #Iterate over all Batches
        Actual_Position_Train=[0,0]
        Max_Position=min(len(Training_Samples[0]),len(Training_Samples[1]))-1
        while max(Actual_Position_Train)<=Max_Position:
            
            Training_Step_Iteration+=1

            #Training Data:
            inputs,labels_raw=GetBatches(Actual_Position_Train,
                                         Training_Data,
                                         Training_Samples,
                                         Seen_Datapoints,
                                         Genom["Repeating_Samples"],
                                         Biasing_Step_Data,
                                         Batchsize)
            labels=torch.Tensor(labels_raw)
            labels=torch.reshape(labels, (-1,1))
            optimizer.zero_grad()

            #Get Embedding
            if Genom["Finetuning_Method"]=="None": #If no finetuning of pretrained Model:
                with torch.no_grad():
                    Tokenized=embedding_model.tokenize(inputs)
                    Tokenized["input_ids"].to(Device_of_Model)
                    Tokenized["attention_mask"].to(Device_of_Model)
                    Embeddings=embedding_model(Tokenized)
            else: #If finetuning of pretrained Model:
                Tokenized=embedding_model.tokenize(inputs)
                Tokenized["input_ids"].to(Device_of_Model)
                Tokenized["attention_mask"].to(Device_of_Model)
                Embeddings=embedding_model(Tokenized)

            #Classification using Embedding
            Classification=adapter_Model(Embeddings["sentence_embedding"].to("cpu"))

            #Save info for repeating hard datapoints
            if Genom["Repeating_Samples"]>0:
                for ij in range(Batchsize):
                    Seen_Datapoints.append([abs(labels_raw[ij]-Classification[ij][0].item())+(Training_Step_Iteration/100),inputs[ij],labels_raw[ij]]) 
                Seen_Datapoints.sort(reverse=True, key= lambda x: x[0])

            #Loss Calculation:
            loss=criterion(Classification,labels) #classification loss
            if Genom["Clustering"]==1: #Bias Loss
                Cluster_Model_Output=clustering_Model(Embeddings["sentence_embedding"].to("cpu"))
                loss_Biasing=Loss_Clustering_Total(Cluster_Model_Output,Genom["Clustering_Loss_to_Center_Loss"])
                loss_ED=loss_embedding_similarity(Embeddings["sentence_embedding"].to("cpu"),labels)
                loss_Biasing=torch.add(torch.mul(loss_Biasing,Genom["Cluster_Loss_to_Distance_Loss"]),torch.mul(loss_ED,(1-Genom["Cluster_Loss_to_Distance_Loss"])))
                loss=torch.add(torch.mul(loss,Genom["Label_Prediction_Loss_to_Biasing_Loss"]),torch.mul(loss_Biasing,(1-Genom["Label_Prediction_Loss_to_Biasing_Loss"])))

            #loss step:
            loss.backward()
            optimizer.step()

            #Save statistics
            if Training_Step_Iteration%math.ceil(Number_Training_Batches/Save_Models_Times)==0:
                eval_term=Eval_New_Model(
                    Test_Data,
                    embedding_model,
                    adapter_Model,
                    Time_To_Evalu,
                    Biasing_Step_Data,
                    Batchsize=8)
                
                loss_statistic[-1].append([loss.item(),eval_term])
                with open ("loss_eval_data.json","w") as f:
                    json.dump(loss_statistic,f)
                file_names=["./local_models/own_models/Embedding_Model_BS_"+str(len(Biasing_Step_Data))+"_TS_"+str(Training_Step_Iteration),
                            "./local_models/own_models/Adapter_Model_BS_"+str(len(Biasing_Step_Data))+"_TS_"+str(Training_Step_Iteration)+".mod",
                            "./local_models/own_models/Clustering_Model_BS_"+str(len(Biasing_Step_Data))+"_TS_"+str(Training_Step_Iteration)+".mod"]
                embedding_model.save(file_names[0])
                torch.save(adapter_Model,file_names[1])
                torch.save(clustering_Model,file_names[2])
                if Training_Step_Iteration>=Number_Training_Batches-1:
                    training_finished=True

                    #Prepare and save Biasing_Step_Data
                    Biasing_Step_Data=Update_Biasing_Step_Data(Biasing_Step_Data,
                                                               embedding_model,
                                                               adapter_Model,
                                                               clustering_Model,
                                                               Biasing_Determining_Time,
                                                               Test_Data,
                                                               file_names)
                    with open("./local_models/own_models/Configuration_Biasing_BS_"+str(len(Biasing_Step_Data)-1)+"_TS_"+str(Training_Step_Iteration)+".json","w") as f:
                        savedata=[]
                        for bsd in Biasing_Step_Data:
                            savedata.append(copy.deepcopy(bsd[1:]))
                            savedata[-1][0]=savedata[-1][0].tolist()
                        json.dump(savedata,f)
                    break
        if training_finished:
            break

    return Biasing_Step_Data




#######################
#    Eval Function    #
#######################

def Eval_New_Model(Test_Data,embedding_model,adapter_Model,Allowed_Time,Biasing_Step_Data,Batchsize=8):
    with torch.no_grad():
        Start_Time_Testing=time.time()
        Test_Samples=MakeBatches(Test_Data,Batchsize)
        Predictions=[]
        True_Labels=[]
        Actual_Position_Test=[0,0]
        Max_Position=min(len(Test_Samples[0]),len(Test_Samples[1]))-1
        while max(Actual_Position_Test)<=Max_Position:
            inputs,labels=GetBatches(Actual_Position_Test,
                                     Test_Data,
                                     Test_Samples,
                                     [],
                                     0,
                                     Biasing_Step_Data,
                                     Batchsize)
            Tokenized=embedding_model.tokenize(inputs)
            Tokenized["input_ids"].to(Device_of_Model)
            Tokenized["attention_mask"].to(Device_of_Model)
            Embeddings=embedding_model(Tokenized)
            Classification=adapter_Model(Embeddings["sentence_embedding"].to("cpu"))
            Classification=torch.reshape(Classification, (-1,))
            Classification=Classification.tolist()
            Predictions+=Classification

            True_Labels+=labels

            if time.time()-Start_Time_Testing>=Allowed_Time:
                break
    Predictions=[ round(elem) for elem in Predictions ]
    return accuracy_score(True_Labels, Predictions)




########################################
#    Data loading and preprocessing    #
########################################


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

def remove_duplicates(ac_Data):
    new_data=[]
    duplicates=0
    for i in tqdm(range(len(ac_Data))):
        non_Duplicate=True
        for j in range(i+1,len(ac_Data)):
            if ac_Data[i]==ac_Data[j]:
                non_Duplicate=False
                duplicates+=1
                break
        if non_Duplicate:
            new_data.append(ac_Data[i])
    return new_data
    

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




####################
#    Used Genom    #
####################

acGenom={}
acGenom["Embedding_Model"]                       = 'intfloat/multilingual-e5-large-instruct'  # Used pretrained Model
acGenom["Finetuning_Method"]                     = "Full"                                     # Finetuning Method
acGenom["Depth_Adapter"]                         = 1                                          # Depth of adapter neural network taking sentence embedding as input and outputs pediction for 0,1 label
acGenom["Width_First_Layer"]                     = nan                                        # Width of first hidden layer
acGenom["Width_Second_Layer"]                    = nan                                        # Width of second hidden layer
acGenom["Learning_Rate_Embedding_Model"]         = 8e-07                                      # Learning Rate pretrained model
acGenom["Learning_Rate_Adapter_Model"]           = 0.001                                      # Learning Rate of adapter neural network
acGenom["Activation_Function_Adapter_Model"]     = nan                                        # Activation Function of adapter Model
acGenom["Lora_R"]                                = nan                                        # LoRA R
acGenom["Lora_Alpha_to_R"]                       = nan                                        # LoRA Alpha relation to Lora R
acGenom["Repeating_Samples"]                     = 0.0                                        # Repeating rate of wrongly classified samples
acGenom["Clustering"]                            = 1                    # Use Clustering Approach (1=yes, 0=no)
acGenom["Learning_Rate_Clustering_Model"]        = 0.001                # Learing Rate clustering neural net
acGenom["Clustering_Loss_to_Center_Loss"]        = 0.90                 # Rate of max clustering loss to cluster center distance loss to get cluster loss (1.0 = only max clustering loss)
acGenom["Cluster_Number"]                        = 14                   # Available clusters
acGenom["K-Means_Initialization"]                = 1                    # Initialize clusters with k-means (1 = yes, 0= no)
acGenom["Label_Prediction_Loss_to_Biasing_Loss"] = 0.70                 # Rate of Label prediction loss to Biasing Loss to get final loss (1.0 = only Label prediction loss)
acGenom["Increase_Cos_Similarity"]               = 0.02                 # Increasing of cosine similarity per batch
acGenom["Decrease_Cos_Similarity"]               = 0.06                 # Decreasing of cosine similarity if cluster was found in batch
acGenom["Cluster_Loss_to_Distance_Loss"]         = 1.0                  # Rate of Cluster loss to Distance loss of different labeled samples to get Biasing Loss (1.0 = only cluster loss)
acGenom["Hierarchical_Steps"]                    = 3                    # How many hierarchical Steps
acGenom["Load_Finetuned_Model"]                  = 1                    # 1 = Use previously trained model in Hierarchy ; 0 = Load New Model each time




####################
#    Bias Steps    #
####################

Biasing_Step_Data=[]
loss_statistic=[]
for aib in range(acGenom["Hierarchical_Steps"]): #Make 3 Biasing steps
    print("*"*100)
    print(aib)
    #print(Biasing_Step_Data)
    loss_statistic.append([])
    Biasing_Step_Data=Train_New_Model(
        Training_Data=(Train_Data_Text_L0,Train_Data_Text_L1),
        Genom=acGenom,
        Number_Training_Batches=Number_Training_Batches,
        Allowed_Time_Cluster_Init=Time_To_Initialize_Clusters,
        Test_Data=(Test_Data_Text_L0,Test_Data_Text_L1),
        loss_statistic=loss_statistic,
        Biasing_Determining_Time=Biasing_Determining_Time,
        Biasing_Step_Data=Biasing_Step_Data,
        Batchsize=8)
    
    if acGenom["Clustering"]==0:
        break

