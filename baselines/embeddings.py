import torch
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np


class Embedding:
    def __init__(self):
        self.name = "Base Embedding"

    def fit(self, tweets):
        pass

    def encode(self, tweets, name=""):
        return tweets

    def __repr__(self):
        return f"{self.name}"


class BagOfWords(Embedding):
    def __init__(self, max_features=5000):
        super().__init__()
        self.name = f"Bag of Words ({max_features})"
        self.vectorizer = CountVectorizer(max_features=max_features)

    def fit(self, tweets):
        self.vectorizer.fit(tweets)

    def encode(self, tweets, name=""):
        return self.vectorizer.transform(tweets)


class RobertaBaseSentimentEmbedding(Embedding):
    def __init__(self, load_embeddings=False):
        super().__init__()
        self.name = "Roberta Base Sentiment"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", torch_dtype=torch.float16).to(self.device)
        self.model.save_pretrained("./models/")
        self.load_embeddings = load_embeddings
        self.embeddings_file = f"embeddings_{self.name.lower().replace(' ', '_')}.npy"

    def encode(self, tweets, name=""):
        if self.load_embeddings:
            try:
                return np.load(name + "_" + self.embeddings_file)
            except FileNotFoundError:
                print("Embeddings file not found. Making new embeddings instead.")

        return self.make_embeddings(tweets, name)

    def make_embeddings(self, tweets, name):
        embeddings = []
        with torch.no_grad():
            for text in tqdm(tweets, desc=f"Encoding {name}"):
                tokens = self.tokenizer(text, padding=True, return_tensors='pt').to(self.device)
                embedding = self.model(**tokens).last_hidden_state[:, 0, :].cpu().detach().numpy().astype(np.float32)
                embeddings.append(embedding)
        np.save(name + "_" + self.embeddings_file, np.vstack(embeddings))
        return np.vstack(embeddings)
