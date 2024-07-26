import json
import random
from itertools import islice

import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

from Training_Model.Bias_is_all_you_need import get_train_and_test_data

NUMBER_OF_ITERATIONS = 100000

_, actual_test_data = get_train_and_test_data()

with open("local_models/own_models/Configuration_Biasing_BS_2_TS_12000.json", "r") as f:
    config_data = json.load(f)

for i in range(len(config_data)):
    config_data[i][0] = torch.as_tensor(config_data[i][0])
    config_data[i].append(SentenceTransformer(config_data[i][2][0]))
    config_data[i].append(torch.load(config_data[i][2][1]))

cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

results = dict()

results["Full_Hierarchy"] = {
    "Total": 0,
    "Correct": 0,
    "Total_0": 0,
    "Correct_0": 0,
    "Total_1": 0,
    "Correct_1": 0,
    "Ensembling_Correct": 0,
    "Ensembling_Correct_0": 0,
    "Ensembling_Correct_1": 0
}

for i in range(len(config_data)):
    results[f"Level_{i}"] = {
        "Total_Seen": 0,
        "Total_Kept": 0,
        "Correct_Seen": 0,
        "Correct_Kept": 0,
        "Total_Seen_0": 0,
        "Total_Kept_0": 0,
        "Correct_Seen_0": 0,
        "Correct_Kept_0": 0,
        "Total_Seen_1": 0,
        "Total_Kept_1": 0,
        "Correct_Seen_1": 0,
        "Correct_Kept_1": 0,
        "Clusters": {}
    }

ac_iteration = 0
while ac_iteration < NUMBER_OF_ITERATIONS:
    print(round(100 * ac_iteration / NUMBER_OF_ITERATIONS), end="\r")
    rl = random.randint(0, 1)
    rt = random.randint(0, len(actual_test_data[rl]) - 1)
    ac_Averaging = []
    for ac_level_number, ac_level in enumerate(config_data):

        try:
            with torch.no_grad():
                Tokenized = ac_level[3].tokenize([actual_test_data[rl][rt]])
                Embeddings = ac_level[3](Tokenized)["sentence_embedding"]
                Prediction_nR = ac_level[4](Embeddings).item()
                Prediction = round(Prediction_nR)
                Embeddings = Embeddings[0]
        except:
            break

        vecs1 = Embeddings.unsqueeze(0).repeat(ac_level[0].shape[0], 1)
        vecs2 = ac_level[0]
        cosSimis = cosine_similarity(vecs1, vecs2)
        maxSimi = torch.max(cosSimis, dim=0)
        acCluster = maxSimi.indices.item()

        ac_Averaging.append(Prediction_nR)

        Cluster_Key = f"Cluster_{acCluster}"
        if Cluster_Key not in results[f"Level_{ac_level_number}"]["Clusters"]:
            results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key] = {}
            results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Biased"] = (acCluster in ac_level[1])
            results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Total"] = 0
            results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Correct"] = 0
            results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Total_0"] = 0
            results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Correct_0"] = 0
            results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Total_1"] = 0
            results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Correct_1"] = 0
            results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Sentences"] = []
        results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Sentences"].append(actual_test_data[rl][rt])
        results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Total"] += 1
        results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Correct"] += 1 - abs(rl - Prediction)
        if rl == 0:
            results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Total_0"] += 1
            results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Correct_0"] += 1 - abs(rl - Prediction)
        else:
            results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Total_1"] += 1
            results[f"Level_{ac_level_number}"]["Clusters"][Cluster_Key]["Correct_1"] += 1 - abs(rl - Prediction)

        results[f"Level_{ac_level_number}"]["Total_Seen"] += 1
        results[f"Level_{ac_level_number}"]["Correct_Seen"] += 1 - abs(rl - Prediction)
        if rl == 0:
            results[f"Level_{ac_level_number}"]["Total_Seen_0"] += 1
            results[f"Level_{ac_level_number}"]["Correct_Seen_0"] += 1 - abs(rl - Prediction)
        else:
            results[f"Level_{ac_level_number}"]["Total_Seen_1"] += 1
            results[f"Level_{ac_level_number}"]["Correct_Seen_1"] += 1 - abs(rl - Prediction)

        if acCluster in ac_level[1] or ac_level_number >= len(config_data) - 1:
            results["Full_Hierarchy"]["Total"] += 1
            results["Full_Hierarchy"]["Correct"] += 1 - abs(rl - Prediction)
            averaged_prediction = round(sum(ac_Averaging) / len(ac_Averaging))
            results["Full_Hierarchy"]["Ensembling_Correct"] += 1 - abs(rl - averaged_prediction)
            if rl == 0:
                results["Full_Hierarchy"]["Total_0"] += 1
                results["Full_Hierarchy"]["Correct_0"] += 1 - abs(rl - Prediction)
                results["Full_Hierarchy"]["Ensembling_Correct_0"] += 1 - abs(rl - averaged_prediction)
            else:
                results["Full_Hierarchy"]["Total_1"] += 1
                results["Full_Hierarchy"]["Correct_1"] += 1 - abs(rl - Prediction)
                results["Full_Hierarchy"]["Ensembling_Correct_1"] += 1 - abs(rl - averaged_prediction)

            results[f"Level_{ac_level_number}"]["Total_Kept"] += 1
            results[f"Level_{ac_level_number}"]["Correct_Kept"] += 1 - abs(rl - Prediction)
            if rl == 0:
                results[f"Level_{ac_level_number}"]["Total_Kept_0"] += 1
                results[f"Level_{ac_level_number}"]["Correct_Kept_0"] += 1 - abs(rl - Prediction)
            else:
                results[f"Level_{ac_level_number}"]["Total_Kept_1"] += 1
                results[f"Level_{ac_level_number}"]["Correct_Kept_1"] += 1 - abs(rl - Prediction)
            ac_iteration += 1
            # print("hey")
            break

if results["Full_Hierarchy"]["Total"] != 0:
    results["Full_Hierarchy"]["Accuracy"] = results["Full_Hierarchy"]["Correct"] / results["Full_Hierarchy"]["Total"]
    results["Full_Hierarchy"]["Accuracy_Ensembling"] = results["Full_Hierarchy"]["Ensembling_Correct"] / results["Full_Hierarchy"]["Total"]
if results["Full_Hierarchy"]["Total_0"] != 0:
    results["Full_Hierarchy"]["Accuracy_0"] = results["Full_Hierarchy"]["Correct_0"] / results["Full_Hierarchy"]["Total_0"]
    results["Full_Hierarchy"]["Accuracy_Ensembling_0"] = results["Full_Hierarchy"]["Ensembling_Correct_0"] / results["Full_Hierarchy"]["Total_0"]
if results["Full_Hierarchy"]["Total_1"] != 0:
    results["Full_Hierarchy"]["Accuracy_1"] = results["Full_Hierarchy"]["Correct_1"] / results["Full_Hierarchy"]["Total_1"]
    results["Full_Hierarchy"]["Accuracy_Ensembling_1"] = results["Full_Hierarchy"]["Ensembling_Correct_1"] / results["Full_Hierarchy"]["Total_1"]

for i in results:
    if i == "Full_Hierarchy":
        continue
    if results[i]["Total_Seen"] != 0:
        results[i]["Accuracy_Seen"] = results[i]["Correct_Seen"] / results[i]["Total_Seen"]
    if results[i]["Total_Seen_0"] != 0:
        results[i]["Accuracy_Seen_0"] = results[i]["Correct_Seen_0"] / results[i]["Total_Seen_0"]
    if results[i]["Total_Seen_1"] != 0:
        results[i]["Accuracy_Seen_1"] = results[i]["Correct_Seen_1"] / results[i]["Total_Seen_1"]
    if results[i]["Total_Kept"] != 0:
        results[i]["Accuracy_Kept"] = results[i]["Correct_Kept"] / results[i]["Total_Kept"]
    if results[i]["Total_Kept_0"] != 0:
        results[i]["Accuracy_Kept_0"] = results[i]["Correct_Kept_0"] / results[i]["Total_Kept_0"]
    if results[i]["Total_Kept_1"] != 0:
        results[i]["Accuracy_Kept_1"] = results[i]["Correct_Kept_1"] / results[i]["Total_Kept_1"]

    for j in results[i]["Clusters"]:
        if results[i]["Clusters"][j]["Total"] != 0:
            results[i]["Clusters"][j]["Accuracy"] = results[i]["Clusters"][j]["Correct"] / results[i]["Clusters"][j]["Total"]
        if results[i]["Clusters"][j]["Total_0"] != 0:
            results[i]["Clusters"][j]["Accuracy_0"] = results[i]["Clusters"][j]["Correct_0"] / results[i]["Clusters"][j]["Total_0"]
        if results[i]["Clusters"][j]["Total_1"] != 0:
            results[i]["Clusters"][j]["Accuracy_1"] = results[i]["Clusters"][j]["Correct_1"] / results[i]["Clusters"][j]["Total_1"]

with open("Determine_Clustering_Results.json", "w") as f:
    json.dump(results, f, sort_keys=False, indent=1)
print(json.dumps(results, sort_keys=False, indent=1), flush=True)
