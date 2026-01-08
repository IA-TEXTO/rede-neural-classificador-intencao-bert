# Classificador de Intenção - IntegraCar

Sistema de classificação de intenções de perguntas utilizando modelo BERT (Bidirectional Encoder Representations from Transformers) treinado em português. O projeto inclui treinamento do modelo, API REST para inferência e containerização com Docker.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Características](#características)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação](#instalação)
- [Uso](#uso)
  - [Treinamento do Modelo](#treinamento-do-modelo)
  - [Executando a API](#executando-a-api)
  - [Docker](#docker)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Endpoints da API](#endpoints-da-api)
- [Exemplos](#exemplos)

## Sobre o Projeto

Este projeto implementa um classificador de intenções baseado em BERT (`neuralmind/bert-base-portuguese-cased`) para análise e categorização automática de perguntas em português. O sistema foi desenvolvido para o projeto IntegraCar e permite classificar perguntas relacionadas a processos administrativos, documentação e sistemas como Simlam e e-Docs.

## Características

-  Modelo BERT pré-treinado em português
-  API REST com FastAPI para inferência em tempo real
-  Containerização com Docker para deployment fácil
-  Treinamento customizável com métricas detalhadas
-  Validação de dados com Pydantic
-  Retorna classificação e nível de confiança
-  Suporte a múltiplas classes de intenção

## Tecnologias Utilizadas

- **Python 3.11**
- **PyTorch 2.9.1** - Framework de deep learning
- **Transformers 4.57.3** - Biblioteca Hugging Face para modelos BERT
- **FastAPI 0.128.0** - Framework web moderno para APIs
- **Uvicorn** - Servidor ASGI de alta performance
- **Pandas 2.2.3** - Manipulação de dados
- **Scikit-learn 1.8.0** - Métricas e pré-processamento
- **NumPy 2.1.0** - Computação numérica

## Instalação

### Pré-requisitos

- Python 3.11 ou superior
- pip (gerenciador de pacotes Python)
- Docker (opcional, para containerização)

### Instalação Local

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### Treinamento do Modelo (OPCIONAL)

Para treinar o modelo com seus próprios dados:

1. Prepare seus dados no formato CSV com as colunas `pergunta` e `classificacao`:
```csv
pergunta,classificacao
Como posso abrir um processo?,Manual
Qual a legislação aplicável?,Legislação
```

2. Execute o script de treinamento:
```bash
python main.py
```

O modelo treinado será salvo na pasta `modelo_treinado/` contendo:
- Pesos do modelo (model.safetensors)
- Tokenizer configurado
- LabelEncoder para as classes
- Metadados de configuração

### Executando a API

#### Modo Local

Execute a API usando Uvicorn:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Lembre-se: A API roda na porta 8000**

Acesse a documentação interativa em: `http://localhost:8000/docs`

#### Modo Docker

Usando a imagem pré-construída do Docker Hub:

```bash
docker pull integracar/redeneuralbert:8.0
docker run -p 8000:8000 integracar/redeneuralbert:8.0
```

Ou construa a imagem localmente:

```bash
docker build -t classificador-bert .
docker run -p 8000:8000 classificador-bert
```

**Importante:** A aplicação roda na porta 8000 dentro do container e deve ser mapeada para a porta desejada no host.

## Estrutura do Projeto

```
rede-neural-classificador-intencao-bert/
│
├── api.py                      # API FastAPI para inferência
├── main.py                     # Script principal de treinamento
├── classificacao.csv           # Dataset de treinamento
├── requirements.txt            # Dependências do projeto
├── Dockerfile                  # Configuração Docker
├── readme.md                   # Este arquivo
│
└── modelo_treinado/            # Modelo treinado (gerado após treinamento)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── labelencoder.pkl
    └── metadata.pkl

```

## 🌐 Endpoints da API

### `GET /`
Retorna informações sobre a API e endpoints disponíveis.

**Resposta:**
```json
{
  "mensagem": "API de Classificação de Intenções com BERT",
  "versao": "1.0.0",
  "documentacao": "/docs",
  "endpoints": {
    "classificar": "/classificar (POST)"
  }
}
```

### `POST /classificar`
Classifica a intenção de uma pergunta.

**Request Body:**
```json
{
  "texto": "Como posso abrir um processo corretamente no Simlam?"
}
```

**Resposta:**
```json
{
  "classe": "Manual",
  "confianca": 0.9534
}
```

## Exemplos

### Exemplo com cURL

```bash
curl -X POST "http://localhost:8000/classificar" \
  -H "Content-Type: application/json" \
  -d '{"texto": "Como a legislação aplicável estabelece os critérios?"}'
```

### Exemplo com Python

```python
import requests

url = "http://localhost:8000/classificar"
data = {
    "texto": "Como posso abrir um processo corretamente no Simlam?"
}

response = requests.post(url, json=data)
resultado = response.json()

print(f"Classe: {resultado['classe']}")
print(f"Confiança: {resultado['confianca']:.2%}")
```

### Exemplo com JavaScript (Fetch API)

```javascript
fetch('http://localhost:8000/classificar', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    texto: 'Como posso abrir um processo corretamente no Simlam?'
  })
})
.then(response => response.json())
.then(data => {
  console.log('Classe:', data.classe);
  console.log('Confiança:', data.confianca);
});
```

## Parâmetros de Configuração

### Classe ClassificadorPerguntasBERT

- `model_name`: Modelo BERT base (padrão: "neuralmind/bert-base-portuguese-cased")
- `max_len`: Comprimento máximo das sequências (padrão: 100)
- `device`: Dispositivo de computação (cuda/cpu, detectado automaticamente)

### Parâmetros de Treinamento

- `epochs`: Número de épocas (padrão: 4)
- `batch_size`: Tamanho do batch (padrão: 10)
- `learning_rate`: Taxa de aprendizado (padrão: 2e-5)
- `test_size`: Proporção dos dados de teste (padrão: 0.3)

## Métricas de Avaliação

O modelo gera automaticamente após o treinamento:
- Matriz de confusão
- Relatório de classificação (precisão, recall, f1-score)
- Loss médio por época

## Docker Hub

O projeto está disponível no Docker Hub:

```bash
docker pull integracar/redeneuralbert:8.0
```

**Nota importante:** A aplicação roda na porta 8000.

## Formato do Dataset

O arquivo `classificacao.csv` deve conter duas colunas:

- `pergunta`: Texto da pergunta a ser classificada
- `classificacao`: Classe/categoria da pergunta

Exemplo:
```csv
pergunta,classificacao
Como posso abrir um processo corretamente no Simlam?,Manual
Quais cuidados devo ter ao protocolar documentos no e-Docs?,Manual
```


