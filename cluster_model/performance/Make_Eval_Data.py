import os
import json
import matplotlib.pyplot as plt
#from transformers import T5Tokenizer, T5ForConditionalGeneration

#tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
#model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")


complist=os.listdir()
complist.sort()

for acComp in complist:
    if acComp[:2]!="cl":
        continue
    with open(acComp+"/loss_eval_data.json") as f:
        loss_eval_data=json.load(f)
    with open(acComp+"/Determine_Clustering_Results.json") as f:
        Clustering_Results=json.load(f)
    print("*"*100)
    print(acComp)
    print("*"*100)
    print()
    print("Accuracy",Clustering_Results["Full_Hierarchy"]["Accuracy"])
    for ackey in Clustering_Results:
        if ackey[:5]=="Level":
            print("Accuracy",ackey,Clustering_Results[ackey]["Accuracy_Kept"])
    print()
    print()
    for ackey in Clustering_Results:
        print(ackey)
        print("-"*100)
        if ackey[:5]=="Level":
            for accluster in Clustering_Results[ackey]["Clusters"]:
                examples=Clustering_Results[ackey]["Clusters"][accluster]["Sentences"][:20]
                inputtext=""#"This are some sample texts forming a cluster:\n\n"
                for actext in examples:
                    inputtext+=actext
                    inputtext+="\n"
                #inputtext+="\n"
                #inputtext+="This cluster can be summarized with the small phrase:"
                #input_ids = tokenizer(inputtext, return_tensors="pt").input_ids

                #outputs = model.generate(input_ids)
                #title=tokenizer.decode(outputs[0])
                print(accluster,Clustering_Results[ackey]["Clusters"][accluster]["Total"])
                print("pos",Clustering_Results[ackey]["Clusters"][accluster]["Total_1"])
                print("neg",Clustering_Results[ackey]["Clusters"][accluster]["Total_0"])
                print("biased",Clustering_Results[ackey]["Clusters"][accluster]["Biased"])
                print(inputtext)
    
    batches_per_level=1200
    if len(loss_eval_data)==1:
        batches_per_level=3600
    for level,ac_loss_arr in enumerate(loss_eval_data):
        accuracy=[]
        batch=[]
        for ac_DP_pos,ac_DP in enumerate(ac_loss_arr):
            accuracy.append(ac_DP[1])
            batch.append(batches_per_level*ac_DP_pos+1)
        plt.figure(figsize=(4,3),dpi=80)
        plt.plot(batch,accuracy)
        #plt.plot(batch,loss,label="loss")
        #plt.legend(loc="best")
        plt.savefig(acComp+"/Level_"+str(level)+"_Accuracy.png")
        plt.clf()



    print()
    print()
    print()
    print()
