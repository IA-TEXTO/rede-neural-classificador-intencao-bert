import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU ativada!")
    except RuntimeError as e:
        print(e)
else:
    print("Nenhuma GPU detectada")

classificacao = pd.read_csv("classificacao.csv")

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(classificacao['classificacao'])
num_classes = len(np.unique(y))

mensagens = classificacao["pergunta"].astype(str).values

X_train, X_test, y_train, y_test = train_test_split(
    mensagens,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


model_name = "neuralmind/bert-base-portuguese-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

max_len = 64


def encode_texts(texts, tokenizer, max_len=64):
    return tokenizer(
        list(texts),
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )


X_train_enc = encode_texts(X_train, tokenizer, max_len)
X_test_enc  = encode_texts(X_test,  tokenizer, max_len)



model = TFAutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    dict(X_train_enc),
    y_train,
    epochs=3,
    batch_size=8,
    validation_data=(dict(X_test_enc), y_test),
    verbose=True,
)



loss_val, accuracy_val = model.evaluate(dict(X_test_enc), y_test, verbose=0)
print("Loss:", loss_val)
print("Acurácia:", accuracy_val)

# Predições para matriz de confusão e classification_report
outputs = model(dict(X_test_enc), training=False)
logits = outputs.logits  # tensor [n_amostras, num_classes]

probs = tf.nn.softmax(logits, axis=-1).numpy()
y_pred_int = probs.argmax(axis=1)

cm = confusion_matrix(y_test, y_pred_int)
print("Matriz de confusão:\n", cm)

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred_int, target_names=labelencoder.classes_))


# =========================
# 6. Função para classificar nova pergunta
# =========================
def classificar_pergunta(texto):
    enc = tokenizer(
        [texto],
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )

    outputs = model(enc, training=False)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

    idx = int(np.argmax(probs))
    classe = labelencoder.inverse_transform([idx])[0]

    return classe, float(probs[idx])



if __name__ == "__main__":
    texto_exemplo = "De que forma uma unidade pode contestar formalmente uma IN que impacta negativamente seus processos internos?"
    classe, confianca = classificar_pergunta(texto_exemplo)
    print(f"Texto: {texto_exemplo}")
    print(f"Classe prevista: {classe} (confiança: {confianca:.4f})")
