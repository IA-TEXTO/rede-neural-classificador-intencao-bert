import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class ClassificadorPerguntasBERT:

    def __init__(self, model_name="neuralmind/bert-base-portuguese-cased", max_len=64, device=None):
        self.model_name = model_name
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.labelencoder = LabelEncoder()
        self.model = None
        self.num_classes = None

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


    def carregar_dados(self, caminho_csv):
        df = pd.read_csv(caminho_csv)

        y = self.labelencoder.fit_transform(df["classificacao"])
        self.num_classes = len(np.unique(y))
        X = df["pergunta"].astype(str).values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )


    def encode(self, texts):
        return self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        )


    def inicializar_modelo(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        ).to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.loss_fn = nn.CrossEntropyLoss()


    def treinar(self, epochs=3, batch_size=8):

        train_enc = self.encode(self.X_train)
        test_enc = self.encode(self.X_test)

        train_dataset = TextDataset(train_enc, self.y_train)
        test_dataset = TextDataset(test_enc, self.y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):

                batch = {k: v.to(self.device) for k, v in batch.items()}

                self.optimizer.zero_grad()

                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Loss médio: {avg_loss:.4f}")


    def avaliar(self):
        enc = self.encode(self.X_test)
        dataset = TextDataset(enc, self.y_test)
        loader = DataLoader(dataset, batch_size=16)

        self.model.eval()
        preds, labels = [], []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                logits = outputs.logits

                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                labels.extend(batch["labels"].cpu().numpy())

        # Métricas
        print("\nMatriz de confusão:")
        print(confusion_matrix(labels, preds))

        print("\nRelatório de classificação:")
        print(classification_report(labels, preds, target_names=self.labelencoder.classes_))


    def classificar(self, texto):
        enc = self.encode([texto])
        batch = {k: torch.tensor(v).to(self.device) for k, v in enc.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        classe = self.labelencoder.inverse_transform([idx])[0]

        return classe, float(probs[idx])


    def salvar_modelo(self, pasta):
        os.makedirs(pasta, exist_ok=True)

        self.model.save_pretrained(pasta)
        self.tokenizer.save_pretrained(pasta)

        with open(os.path.join(pasta, "labelencoder.pkl"), "wb") as f:
            pickle.dump(self.labelencoder, f)

        meta = {
            "model_name": self.model_name,
            "max_len": self.max_len,
            "num_classes": self.num_classes,
        }
        with open(os.path.join(pasta, "metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)

        print(f"Modelo salvo em {pasta}")


    def carregar_modelo(self, pasta):
        self.tokenizer = AutoTokenizer.from_pretrained(pasta)

        with open(os.path.join(pasta, "labelencoder.pkl"), "rb") as f:
            self.labelencoder = pickle.load(f)

        with open(os.path.join(pasta, "metadata.pkl"), "rb") as f:
            meta = pickle.load(f)

        self.model_name = meta["model_name"]
        self.max_len = meta["max_len"]
        self.num_classes = meta["num_classes"]

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pasta,
            num_labels=self.num_classes
        ).to(self.device)

        print(f"Modelo carregado de {pasta}")


if __name__ == "__main__":
    clf = ClassificadorPerguntasBERT()

    clf.carregar_dados("classificacao.csv")
    clf.inicializar_modelo()
    clf.treinar(epochs=3, batch_size=8)
    clf.avaliar()

    clf.salvar_modelo("modelo")

    texto = input("Insira a pergunta desejada: ")
    classe, confianca = clf.classificar(texto)

    print("\nPergunta:", texto)
    print("Classe:", classe)
    print("Confiança:", confianca)
