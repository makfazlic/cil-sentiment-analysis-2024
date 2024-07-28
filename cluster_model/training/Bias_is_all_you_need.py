###################
#    Libraries    #
###################

import copy
import json
import math
import random
import sys
import time
import numpy as np
import torch

from collections import defaultdict
from itertools import islice
from numpy import nan
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

###################################
#    Train and Eval Parameters    #
###################################

MINUTE = 60
EVALUATION_TIME = 15 * MINUTE
CLUSTER_INITIALIZATION_TIME = 30 * MINUTE
BIASING_DETERMINING_TIME = 30 * MINUTE
DEVICE = "cpu"
NUMBER_OF_TRAINING_BATCHES = 6000 * 2
NUMBER_OF_SAVED_MODELS = 10


#####################
#    Neural Nets    #
#####################

class AdaptorNet(nn.Module):
    def __init__(self, genome, input_size):
        super(AdaptorNet, self).__init__()
        layer_list = []
        for i in range(genome["Depth_Adapter"]):
            layer_in_size = None
            if i == 0:
                layer_in_size = input_size
            elif i == 1:
                layer_in_size = genome["Width_First_Layer"]
            elif i == 2:
                layer_in_size = genome["Width_Second_Layer"]

            layer_out_size = None
            if i == genome["Depth_Adapter"] - 1:
                layer_out_size = 1
            elif i == 0:
                layer_out_size = genome["Width_First_Layer"]
            elif i == 1:
                layer_out_size = genome["Width_Second_Layer"]
            layer_list.append(nn.Linear(layer_in_size, layer_out_size))
        self.layer_list = nn.ModuleList(layer_list)

        self.activation_function = None
        if genome["Activation_Function_Adapter_Model"] == "ReLU":
            self.activation_function = nn.ReLU()
        elif genome["Activation_Function_Adapter_Model"] == "Sigmoid":
            self.activation_function = nn.Sigmoid()
        elif genome["Activation_Function_Adapter_Model"] == "Tanh":
            self.activation_function = nn.Tanh()

        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        for layer in self.layer_list[:-1]:
            data = layer(data)
            data = self.activation_function(data)
        data = self.layer_list[-1](data)
        return self.sigmoid(data)


class ClusteringNet(nn.Module):
    def __init__(self, embedding_dim, initialization, adapt_plus_percentage, adapt_minus_percentage, num_of_clusters=10):
        super(ClusteringNet, self).__init__()

        self.adapt_plus_percentage = adapt_plus_percentage
        self.adapt_minus_percentage = adapt_minus_percentage
        self.num_of_clusters = num_of_clusters
        self.norm_constant = self.num_of_clusters * self.num_of_clusters - self.num_of_clusters
        if initialization is None:
            self.cluster_means = nn.Parameter(torch.rand((num_of_clusters, embedding_dim), requires_grad=True))
        else:
            self.cluster_means = nn.Parameter(torch.tensor(initialization, requires_grad=True))
        self.max_modifier = torch.zeros(num_of_clusters, requires_grad=False)

        self.cosine_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, data):
        vecs1 = data.unsqueeze(1).repeat(1, self.num_of_clusters, 1)
        vecs2 = self.cluster_means.unsqueeze(0).repeat(vecs1.shape[0], 1, 1)

        cosine_similarities = self.cosine_similarity(vecs1, vecs2)
        max_modifier = self.max_modifier.unsqueeze(0).repeat(vecs1.shape[0], 1)
        cosine_similarities_adapted = torch.add(cosine_similarities, max_modifier)

        max_cosine_similarity = torch.max(cosine_similarities_adapted, dim=1)  # .values
        max_cosine_similarity_adapted = torch.sub(max_cosine_similarity.values, self.max_modifier[max_cosine_similarity.indices])

        vecs3 = self.cluster_means.unsqueeze(0).repeat(self.num_of_clusters, 1, 1)
        vecs4 = self.cluster_means.unsqueeze(1).repeat(1, self.num_of_clusters, 1)
        cosine_similarities_cc = self.cosine_similarity(vecs3, vecs4)

        cosine_similarities_cc.fill_diagonal_(0.0)
        cosine_similarities_cc_sum = torch.sum(cosine_similarities_cc)
        cosine_similarities_cc_sum_normalized = torch.div(cosine_similarities_cc_sum, self.norm_constant)

        max_difference = (torch.max(cosine_similarities) - torch.min(cosine_similarities)).item()
        self.max_modifier = torch.add(self.max_modifier, self.adapt_plus_percentage * max_difference)
        self.max_modifier[max_cosine_similarity.indices] -= (self.adapt_minus_percentage * max_difference)
        self.max_modifier = torch.maximum(self.max_modifier, torch.zeros_like(self.max_modifier))
        self.max_modifier = torch.minimum(self.max_modifier, torch.full_like(self.max_modifier, 2))

        return max_cosine_similarity_adapted, cosine_similarities_cc_sum_normalized


########################
#    Batch Creation    #
########################

def make_batches(data, batchsize):
    pos_list_l0 = list(range(len(data[0])))
    pos_list_l1 = list(range(len(data[1])))
    random.shuffle(pos_list_l0)
    random.shuffle(pos_list_l1)
    return pos_list_l0, pos_list_l1


def datapoint_is_biased(ac_datapoint, biasing_step_data):
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    with torch.no_grad():
        for biasing_step_data_point in biasing_step_data:
            tokenized = biasing_step_data_point[0].tokenize([ac_datapoint])
            tokenized["input_ids"].to(DEVICE)
            tokenized["attention_mask"].to(DEVICE)
            embeddings = biasing_step_data_point[0](tokenized)["sentence_embedding"].to("cpu")[0]
            vecs1 = embeddings.unsqueeze(0).repeat(biasing_step_data_point[1].shape[0], 1)
            vecs2 = biasing_step_data_point[1]
            cosine_similarities = cosine_similarity(vecs1, vecs2)
            max_similarity = torch.max(cosine_similarities, dim=0)
            ac_cluster = max_similarity.indices.item()
            if ac_cluster in biasing_step_data_point[2]:
                return True
    return False


def get_batches(actual_position, training_data, training_samples, seen_datapoints, take_seen_one_prob, biasing_step_data, batch_size):
    if batch_size % 2 != 0:
        sys.exit("Batch has to be divisible by 2")

    half_of_batch_size = batch_size // 2
    batch = [[], []]

    if random.random() < take_seen_one_prob and len(seen_datapoints) > 10:
        for _ in range(batch_size):
            length_seen_datapoints = len(seen_datapoints)
            position_inserting_datapoint = length_seen_datapoints - 1
            for _ in range(max(1, length_seen_datapoints // 5)):
                position_inserting_datapoint = min(position_inserting_datapoint, random.randint(0, length_seen_datapoints - 1))
            batch[0].append(seen_datapoints[position_inserting_datapoint][1])
            batch[1].append(seen_datapoints[position_inserting_datapoint][2])
            del seen_datapoints[position_inserting_datapoint]
    else:
        neg_samples = 0
        while neg_samples < half_of_batch_size and actual_position[0] < len(training_samples[0]):
            if not datapoint_is_biased(training_data[0][training_samples[0][actual_position[0]]], biasing_step_data):
                neg_samples += 1
                batch[0].append(training_data[0][training_samples[0][actual_position[0]]])
                batch[1].append(0)
            actual_position[0] += 1
        pos_samples = 0
        while pos_samples < half_of_batch_size and actual_position[1] < len(training_samples[1]):
            if not datapoint_is_biased(training_data[1][training_samples[1][actual_position[1]]], biasing_step_data):
                pos_samples += 1
                batch[0].append(training_data[1][training_samples[1][actual_position[1]]])
                batch[1].append(1)
            actual_position[1] += 1

    return batch


########################
#    Loss Functions    #
########################

mse_loss = nn.MSELoss()


def loss_max_clustering(output):
    return mse_loss(output, torch.ones_like(output))


def loss_cluster_distance(output):
    return mse_loss(output, torch.zeros_like(output))


def loss_clustering_total(output, alpha):
    loss_1 = loss_max_clustering(output[0])
    loss_1 = torch.mul(loss_1, alpha)

    loss_2 = loss_cluster_distance(output[1])
    loss_2 = torch.mul(loss_2, (1 - alpha))

    return torch.add(loss_1, loss_2)


def loss_embedding_similarity(embeddings, labels):
    cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
    les_loss = 0
    number_of_elements_l = 0
    for i_1 in range(embeddings.shape[0]):
        for i_2 in range(i_1 + 1, embeddings.shape[0]):
            if labels[i_1][0] != labels[i_2][0]:
                les_loss = torch.add(cosine_similarity(embeddings[i_1], embeddings[i_2]), les_loss)

                number_of_elements_l += 1
    les_loss = torch.div(les_loss, number_of_elements_l)
    les_loss = torch.add(les_loss, 1)

    return les_loss


#################################
#    Biasing Helper Function    #
#################################


def update_biasing_step_data(biasing_step_data, embedding_model, adapter_model, clustering_model, biasing_determining_time, biasing_data, file_names):
    CosineSimilarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    cluster_accuracy_pl = defaultdict(lambda: {"Correct": 0, "Total": 0})
    cluster_biased = []
    clustering_centers = clustering_model.cluster_means.data
    with torch.no_grad():

        start_time_biasing = time.time()
        all_datapoints_biased = 0
        while time.time() - start_time_biasing < biasing_determining_time:
            rl = random.randint(0, 1)
            rt = random.randint(0, len(biasing_data[rl]) - 1)
            if datapoint_is_biased(biasing_data[rl][rt], biasing_step_data):
                continue
            try:
                with torch.no_grad():
                    embeddings = get_embeddings(embedding_model, [biasing_data[rl][rt]])["sentence_embedding"].to("cpu")
                    prediction_nR = adapter_model(embeddings).item()
                    prediction = round(prediction_nR)
                    embeddings = embeddings[0]
            except:
                continue
            vecs1 = embeddings.unsqueeze(0).repeat(clustering_centers.shape[0], 1)
            vecs2 = clustering_centers
            cosine_similarities = CosineSimilarity(vecs1, vecs2)
            maximum_similarity = torch.max(cosine_similarities, dim=0)
            cluster = maximum_similarity.indices.item()

            all_datapoints_biased += 1
            cluster_accuracy_pl[cluster]["Total"] += 1
            if prediction == rl:
                cluster_accuracy_pl[cluster]["Correct"] += 1

        cluster_value_s10 = []
        for key in cluster_accuracy_pl:
            cluster_value_s10.append([key, cluster_accuracy_pl[key]["Correct"] / cluster_accuracy_pl[key]["Total"]])
        cluster_value_s10.sort(reverse=True, key=lambda x: x[1])

        accounted_datapoints_biased = 0
        for key, _ in cluster_value_s10:
            if accounted_datapoints_biased < all_datapoints_biased / 2:
                cluster_biased.append(key)
                accounted_datapoints_biased += cluster_accuracy_pl[key]["Total"]
            else:
                break
    biasing_step_data.append([])
    biasing_step_data[-1].append(copy.deepcopy(embedding_model.to("cpu")).to(DEVICE))
    biasing_step_data[-1].append(copy.deepcopy(clustering_model.cluster_means.data))
    biasing_step_data[-1].append(copy.deepcopy(cluster_biased))
    biasing_step_data[-1].append(copy.deepcopy(file_names))

    return biasing_step_data


########################
#    Train Function    #
########################


def train_new_model(train_data,
                    genome,
                    number_training_batches,
                    allowed_time_cluster_init,
                    test_data,
                    biasing_step_data,
                    loss_statistic,
                    biasing_determining_time,
                    batch_size=8):
    # Statistics
    seen_datapoints = []

    # Load pretrained data
    if len(biasing_step_data) == 0 or genome["Load_Finetuned_Model"] == 0:
        embedding_model = SentenceTransformer(genome["Embedding_Model"], device=DEVICE)
    else:
        embedding_model = SentenceTransformer(biasing_step_data[-1][3][0], device=DEVICE)

    # Extract Size of embedding
    with torch.no_grad():
        examples_embedding_size = embedding_model.encode(["Testing"]).shape[1]

    # Prepare Adaptor Model
    if len(biasing_step_data) == 0 or genome["Load_Finetuned_Model"] == 0:
        adapter_model = AdaptorNet(genome, examples_embedding_size)
    else:
        adapter_model = torch.load(biasing_step_data[-1][3][1])

    # Make K-means Initialization if needed
    clustering_model = None
    if genome["Clustering"] == 1:
        if genome["K-Means_Initialization"] == 0:
            clustering_model = ClusteringNet(embedding_dim=examples_embedding_size,
                                             initialization=None,
                                             adapt_plus_percentage=genome["Increase_Cos_Similarity"],
                                             adapt_minus_percentage=genome["Decrease_Cos_Similarity"],
                                             num_of_clusters=genome["Cluster_Number"])
        else:
            pos_embeddings = []
            neg_embeddings = []
            start_time_clustering_init = time.time()

            while allowed_time_cluster_init > time.time() - start_time_clustering_init:
                ac_sample_pos = random.randint(0, len(train_data[0]) - 1)
                if random.random() > 0.5:
                    ac_sample_text = train_data[0][ac_sample_pos]
                    ac_sample_label = 0
                else:
                    ac_sample_text = train_data[1][ac_sample_pos]
                    ac_sample_label = 1
                if datapoint_is_biased(ac_sample_text, biasing_step_data):
                    continue
                with torch.no_grad():
                    embeddings = get_embeddings(embedding_model, [ac_sample_text])["sentence_embedding"][0].tolist()
                if ac_sample_label < 0.5:
                    neg_embeddings.append(embeddings)
                else:
                    pos_embeddings.append(embeddings)
            number_of_clusters = round(genome["Cluster_Number"] / 2)
            clustering_init_pos = []
            for ac_cluster_num in range(number_of_clusters, 0, -1):
                try:
                    kmeans = KMeans(n_clusters=ac_cluster_num)
                    kmeans.fit(pos_embeddings)
                    clustering_init_pos = list(kmeans.cluster_centers_)
                    break
                except:
                    print("[Error] Pos Clustering not possible", flush=True)
            clustering_init_neg = []
            for ac_cluster_num in range(number_of_clusters, 0, -1):
                try:
                    kmeans = KMeans(n_clusters=ac_cluster_num)
                    kmeans.fit(neg_embeddings)
                    clustering_init_neg = list(kmeans.cluster_centers_)
                    break
                except:
                    print("[Error] Neg Clustering not possible", flush=True)
            clustering_init = clustering_init_pos + clustering_init_neg
            if len(clustering_init) < genome["Cluster_Number"]:
                additional_mean = np.random.rand(genome["Cluster_Number"] - len(clustering_init), examples_embedding_size).tolist()
                clustering_init += additional_mean
            clustering_model = ClusteringNet(embedding_dim=examples_embedding_size,
                                             initialization=clustering_init,
                                             adapt_plus_percentage=genome["Increase_Cos_Similarity"],
                                             adapt_minus_percentage=genome["Decrease_Cos_Similarity"],
                                             num_of_clusters=genome["Cluster_Number"])

    # Prepare Learning rates and optimizer
    if genome["Finetuning_Method"] == "None" and genome["Clustering"] == 0:
        optimizer = torch.optim.Adam([
            {'params': adapter_model.parameters(), 'lr': genome["Learning_Rate_Adapter_Model"]}
        ], lr=genome["Learning_Rate_Adapter_Model"])
    elif genome["Finetuning_Method"] == "None" and genome["Clustering"] == 1:
        optimizer = torch.optim.Adam([
            {'params': adapter_model.parameters(), 'lr': genome["Learning_Rate_Adapter_Model"]},
            {'params': clustering_model.parameters(), 'lr': genome["Learning_Rate_Clustering_Model"]}
        ], lr=genome["Learning_Rate_Adapter_Model"])
    elif genome["Clustering"] == 0:
        optimizer = torch.optim.Adam([
            {'params': embedding_model.parameters(), 'lr': genome["Learning_Rate_Embedding_Model"]},
            {'params': adapter_model.parameters(), 'lr': genome["Learning_Rate_Adapter_Model"]}
        ], lr=genome["Learning_Rate_Embedding_Model"])
    else:
        optimizer = torch.optim.Adam([
            {'params': embedding_model.parameters(), 'lr': genome["Learning_Rate_Embedding_Model"]},
            {'params': adapter_model.parameters(), 'lr': genome["Learning_Rate_Adapter_Model"]},
            {'params': clustering_model.parameters(), 'lr': genome["Learning_Rate_Clustering_Model"]}
        ], lr=genome["Learning_Rate_Embedding_Model"])

    # Training loop
    criterion = nn.BCELoss()
    training_step_iteration = -1
    training_finished = False
    while not training_finished:

        # Prepare Batches:
        training_samples = make_batches(train_data, batch_size)

        # Iterate over all Batches
        actual_position_train = [0, 0]
        max_position = min(len(training_samples[0]), len(training_samples[1])) - 1
        while max(actual_position_train) <= max_position:

            training_step_iteration += 1

            # Training Data:
            inputs, labels_raw = get_batches(actual_position_train,
                                             train_data,
                                             training_samples,
                                             seen_datapoints,
                                             genome["Repeating_Samples"],
                                             biasing_step_data,
                                             batch_size)
            labels = torch.Tensor(labels_raw)
            labels = torch.reshape(labels, (-1, 1))
            optimizer.zero_grad()

            # Get Embedding
            if genome["Finetuning_Method"] == "None":
                with torch.no_grad():
                    embeddings = get_embeddings(embedding_model, inputs)
            else:
                embeddings = get_embeddings(embedding_model, inputs)

            # Classification using Embedding
            classification = adapter_model(embeddings["sentence_embedding"].to("cpu"))

            # Save info for repeating hard datapoints
            if genome["Repeating_Samples"] > 0:
                for ij in range(batch_size):
                    seen_datapoints.append(
                        [abs(labels_raw[ij] - classification[ij][0].item()) + (training_step_iteration / 100), inputs[ij], labels_raw[ij]])
                seen_datapoints.sort(reverse=True, key=lambda x: x[0])

            # Loss Calculation:
            loss = criterion(classification, labels)  # classification loss
            if genome["Clustering"] == 1:  # Bias Loss
                cluster_model_output = clustering_model(embeddings["sentence_embedding"].to("cpu"))
                loss_biasing = loss_clustering_total(cluster_model_output, genome["Clustering_Loss_to_Center_Loss"])
                loss_ed = loss_embedding_similarity(embeddings["sentence_embedding"].to("cpu"), labels)
                loss_biasing = torch.add(torch.mul(loss_biasing, genome["Cluster_Loss_to_Distance_Loss"]),
                                         torch.mul(loss_ed, (1 - genome["Cluster_Loss_to_Distance_Loss"])))
                loss = torch.add(torch.mul(loss, genome["Label_Prediction_Loss_to_Biasing_Loss"]),
                                 torch.mul(loss_biasing, (1 - genome["Label_Prediction_Loss_to_Biasing_Loss"])))

            # loss step:
            loss.backward()
            optimizer.step()

            # Save statistics
            if training_step_iteration % math.ceil(number_training_batches / NUMBER_OF_SAVED_MODELS) == 0:
                eval_term = evaluate_new_model(
                    test_data,
                    embedding_model,
                    adapter_model,
                    EVALUATION_TIME,
                    biasing_step_data,
                    batch_size=8)

                loss_statistic[-1].append([loss.item(), eval_term])
                with open("loss_eval_data.json", "w") as f:
                    json.dump(loss_statistic, f)
                file_names = [f"./local_models/own_models/Embedding_Model_BS_{len(biasing_step_data)}_TS_{training_step_iteration}",
                              f"./local_models/own_models/Adapter_Model_BS_{len(biasing_step_data)}_TS_{training_step_iteration}.mod",
                              f"./local_models/own_models/Clustering_Model_BS_{len(biasing_step_data)}_TS_{training_step_iteration}.mod"]
                embedding_model.save(file_names[0])
                torch.save(adapter_model, file_names[1])
                torch.save(clustering_model, file_names[2])
                if training_step_iteration >= number_training_batches - 1:
                    training_finished = True

                    # Prepare and save Biasing_Step_Data
                    biasing_step_data = update_biasing_step_data(biasing_step_data,
                                                                 embedding_model,
                                                                 adapter_model,
                                                                 clustering_model,
                                                                 biasing_determining_time,
                                                                 test_data,
                                                                 file_names)
                    with open(f"./local_models/own_models/Configuration_Biasing_BS_{len(biasing_step_data) - 1}_TS_{training_step_iteration}.json",
                              "w") as f:
                        savedata = []
                        for bsd in biasing_step_data:
                            savedata.append(copy.deepcopy(bsd[1:]))
                            savedata[-1][0] = savedata[-1][0].tolist()
                        json.dump(savedata, f)
                    break

    return biasing_step_data


def get_embeddings(embedding_model, texts):
    tokenized = embedding_model.tokenize(texts)
    tokenized["input_ids"] = tokenized["input_ids"].to(DEVICE)
    tokenized["attention_mask"] = tokenized["attention_mask"].to(DEVICE)
    return embedding_model(tokenized)


#######################
#    Eval Function    #
#######################

def evaluate_new_model(test_data, embedding_model, adapter_model, allowed_time, biasing_step_data, batch_size=8):
    with torch.no_grad():
        start_time_testing = time.time()
        test_samples = make_batches(test_data, batch_size)
        predictions = []
        true_labels = []
        actual_position_test = [0, 0]
        max_position = min(len(test_samples[0]), len(test_samples[1])) - 1
        while max(actual_position_test) <= max_position:
            inputs, labels = get_batches(actual_position_test,
                                         test_data,
                                         test_samples,
                                         [],
                                         0,
                                         biasing_step_data,
                                         batch_size)
            embeddings = get_embeddings(embedding_model, inputs)
            classification = adapter_model(embeddings["sentence_embedding"].to("cpu"))
            classification = torch.reshape(classification, (-1,))
            classification = classification.tolist()
            predictions += classification

            true_labels += labels

            if time.time() - start_time_testing >= allowed_time:
                break
    predictions = [round(elem) for elem in predictions]
    return accuracy_score(true_labels, predictions)


########################################
#    Data loading and preprocessing    #
########################################

def get_train_and_test_data():
    def matches_pattern(line):
        return "<url>" in line and (line.count("(") > line.count(")"))

    def load_data(filename):

        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            count = 1250000
            for line in tqdm(islice(f, count), total=count, desc='Loading Tweets'):
                line = line.rstrip()
                if not matches_pattern(line) and line not in data:
                    data.append(line)
        return data

    input_data = {0: load_data("./train_neg_full.txt"),
                  1: load_data("./train_pos_full.txt")}

    train_data_labeled0, test_data_labeled0 = train_test_split(input_data[0], test_size=0.33, random_state=42)
    train_data_labeled1, test_data_labeled1 = train_test_split(input_data[1], test_size=0.33, random_state=42)
    return (train_data_labeled0, train_data_labeled1), (test_data_labeled0, test_data_labeled1)


#####################
#    Used Genome    #
#####################

actual_genome = {
    "Embedding_Model": 'intfloat/multilingual-e5-large-instruct',  # Used pretrained Model
    "Finetuning_Method": "Full",  # Finetuning Method
    "Depth_Adapter": 1,  # Depth of adapter neural network taking sentence embedding as input and outputs prediction for 0,1 label
    "Width_First_Layer": nan,  # Width of first hidden layer
    "Width_Second_Layer": nan,  # Width of second hidden layer
    "Learning_Rate_Embedding_Model": 8e-07,  # Learning Rate pretrained model
    "Learning_Rate_Adapter_Model": 0.001,  # Learning Rate of adapter neural network
    "Activation_Function_Adapter_Model": nan,  # Activation Function of adapter Model
    "Lora_R": nan,  # LoRA R
    "Lora_Alpha_to_R": nan,  # LoRA Alpha relation to Lora R
    "Repeating_Samples": 0.0,  # Repeating rate of wrongly classified samples
    "Clustering": 1,  # Use Clustering Approach (1=yes, 0=no)
    "Learning_Rate_Clustering_Model": 0.001,  # Learning Rate clustering neural net
    "Clustering_Loss_to_Center_Loss": 0.90,
    # Rate of max clustering loss to cluster center distance loss to get cluster loss (1.0: only max clustering loss)
    "Cluster_Number": 14,  # Available clusters
    "K-Means_Initialization": 1,  # Initialize clusters with k-means (1: yes, 0= no)
    "Label_Prediction_Loss_to_Biasing_Loss": 0.70,
    # Rate of Label prediction loss to Biasing Loss to get final loss (1.0: only Label prediction loss)
    "Increase_Cos_Similarity": 0.02,  # Increasing of cosine similarity per batch
    "Decrease_Cos_Similarity": 0.06,  # Decreasing of cosine similarity if cluster was found in batch
    "Cluster_Loss_to_Distance_Loss": 1.0,
    # Rate of Cluster loss to Distance loss of different labeled samples to get Biasing Loss (1.0: only cluster loss)
    "Hierarchical_Steps": 3,  # How many hierarchical Steps
    "Load_Finetuned_Model": 1  # 1 = Use previously trained model in Hierarchy ; 0 = Load New Model each time
}

####################
#    Bias Steps    #
####################

actual_train_data, actual_test_data = get_train_and_test_data()

actual_biasing_step_data = []
loss_statistic = []
for aib in range(actual_genome["Hierarchical_Steps"]):  # Make 3 Biasing steps
    print("*" * 100)
    print(aib)

    loss_statistic.append([])
    actual_biasing_step_data = train_new_model(
        train_data=actual_train_data,
        genome=actual_genome,
        number_training_batches=NUMBER_OF_TRAINING_BATCHES,
        allowed_time_cluster_init=CLUSTER_INITIALIZATION_TIME,
        test_data=actual_test_data,
        loss_statistic=loss_statistic,
        biasing_determining_time=BIASING_DETERMINING_TIME,
        biasing_step_data=actual_biasing_step_data,
        batch_size=8)

    if actual_genome["Clustering"] == 0:
        break
