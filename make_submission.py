import csv
import json
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from tqdm import tqdm


def load_data():
    lines = []
    with open("./data/test_data.txt", 'r', encoding='utf-8') as infile:
        for line in infile.read().split("\n")[:-1]:
            assert line.count(",") > 0
            _, line = line.split(",", 1)
            lines.append(line)
    return lines


texts = load_data()
with open('./data/sample_submission.csv', 'r') as f:
    predictions = list(csv.reader(f, delimiter=","))

with open("./work_files/local_models/own_models/Configuration_Biasing_BS_2_TS_12000.json", "r") as f:
    config_data = json.load(f)

for i in range(len(config_data)):
    config_data[i][0] = torch.as_tensor(config_data[i][0])
    config_data[i].append(SentenceTransformer(config_data[i][2][0]))
    config_data[i].append(torch.load(config_data[i][2][1]))


def get_prediction(ac_text):
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    if "<url>" in ac_text and (ac_text.count("(") > ac_text.count(")")):
        return 0, -1
    for ac_level_number, ac_level in enumerate(config_data):
        try:
            with torch.no_grad():
                tokenized = ac_level[3].tokenize([ac_text])
                embeddings = ac_level[3](tokenized)["sentence_embedding"]
                prediction_nr = ac_level[4](embeddings).item()
                prediction = round(prediction_nr)
                embeddings = embeddings[0]
        except:
            return 1, -1

        vecs1 = embeddings.unsqueeze(0).repeat(ac_level[0].shape[0], 1)
        vecs2 = ac_level[0]
        cosSimis = cosine_similarity(vecs1, vecs2)
        maxSimi = torch.max(cosSimis, dim=0)
        cluster = maxSimi.indices.item()

        if cluster in ac_level[1] or ac_level_number >= len(config_data) - 1:
            return 2 + ac_level_number, prediction * 2 - 1


prediction_share = [0, 0, 0, 0, 0]
for index, text in enumerate(tqdm(texts)):
    share_index, prediction = get_prediction(text)
    prediction_share[share_index] += 1
    predictions[index + 1][1] = prediction

submission_filepath = './submission/my_submission.csv'
os.makedirs(os.path.dirname(submission_filepath), exist_ok=True)
with open(submission_filepath, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(predictions)

print(prediction_share, flush=True)
