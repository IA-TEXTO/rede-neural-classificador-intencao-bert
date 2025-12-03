import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


class ClassificadorPerguntasBERT:

    def __init__(self, model_name="neuralmind/bert-base-portuguese-cased", max_len=64):
        self.model_name = model_name
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.labelencoder = LabelEncoder()
        self.model = None
        self.num_classes = None


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
            max_length=self.max_len,
            return_tensors="tf"
        )


    def inicializar_modelo(self):
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])


    def treinar(self, epochs=3, batch_size=8):
        X_train_enc = self.encode(self.X_train)
        X_test_enc = self.encode(self.X_test)

        self.history = self.model.fit(
            dict(X_train_enc),
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(dict(X_test_enc), self.y_test),
            verbose=True
        )


    def avaliar(self):
        X_test_enc = self.encode(self.X_test)

        loss, accuracy = self.model.evaluate(dict(X_test_enc), self.y_test, verbose=0)

        print(f"Loss: {loss}")
        print(f"Acurácia: {accuracy}")

        outputs = self.model(dict(X_test_enc), training=False)
        logits = outputs.logits
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        y_pred = probs.argmax(axis=1)

        print("\nMatriz de Confusão:")
        print(confusion_matrix(self.y_test, y_pred))

        print("\nRelatório de Classificação:")
        print(classification_report(self.y_test, y_pred, target_names=self.labelencoder.classes_))


    def classificar(self, texto):
        enc = self.encode([texto])
        outputs = self.model(enc, training=False)
        probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]

        idx = int(np.argmax(probs))
        classe = self.labelencoder.inverse_transform([idx])[0]

        return classe, float(probs[idx])

    def salvar_modelo(self, pasta):
        os.makedirs(pasta, exist_ok=True)

        self.model.save_pretrained(pasta)

        self.tokenizer.save_pretrained(pasta)

        with open(os.path.join(pasta, "labelencoder.pkl"), "wb") as f:
            pickle.dump(self.labelencoder, f)

        # Salvar metadados
        meta = {
            "model_name": self.model_name,
            "max_len": self.max_len,
            "num_classes": self.num_classes,
        }
        with open(os.path.join(pasta, "metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)

        print(f"Modelo salvo em: {pasta}")


    def carregar_modelo(self, pasta):
        # Carregar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pasta)

        # Carregar labelencoder
        with open(os.path.join(pasta, "labelencoder.pkl"), "rb") as f:
            self.labelencoder = pickle.load(f)

        # Carregar metadados
        with open(os.path.join(pasta, "metadata.pkl"), "rb") as f:
            meta = pickle.load(f)

        self.model_name = meta["model_name"]
        self.max_len = meta["max_len"]
        self.num_classes = meta["num_classes"]

        # Carregar modelo
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            pasta,
            num_labels=self.num_classes
        )

        print(f"Modelo carregado de: {pasta}")


if __name__ == "__main__":
    clf = ClassificadorPerguntasBERT()

    clf.carregar_dados("classificacao.csv")
    clf.inicializar_modelo()
    clf.treinar(epochs=3, batch_size=8)
    clf.avaliar()

    clf.salvar_modelo("modelo_treinado")

    texto = "De que forma uma unidade pode contestar formalmente uma IN?"
    classe, confianca = clf.classificar(texto)

    print("\nPergunta:", texto)
    print("Classe:", classe)
    print("Confiança:", confianca)



    
