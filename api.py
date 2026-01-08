from fastapi import FastAPI
from pydantic import BaseModel, Field
from main import ClassificadorPerguntasBERT

app = FastAPI(
    title="API de Classificação de Intenções - BERT",
    description="API para classificar intenções de perguntas usando modelo BERT treinado",
    version="1.0.0"
)

clf = ClassificadorPerguntasBERT()
clf.carregar_modelo("modelo_treinado")


class TextoEntrada(BaseModel):
    texto: str = Field(
        ...,
        description="Texto da pergunta a ser classificada",
        json_schema_extra={
            "example": "Como a legislação aplicável estabelece os critérios e os limites que devem ser observados nos procedimentos internos de tramitação de processos administrativos?"
        },
    )


class ClassificacaoResposta(BaseModel):
    classe: str = Field(..., description="Classe/intenção identificada")
    confianca: float = Field(..., description="Nível de confiança da classificação (0-1)")


@app.get("/")
async def root():
    return {
        "mensagem": "API de Classificação de Intenções com BERT",
        "versao": "1.0.0",
        "documentacao": "/docs",
        "endpoints": {
            "classificar": "/classificar (POST)"
        }
    }

@app.post("/classificar", response_model=ClassificacaoResposta)
async def classificar(dados: TextoEntrada):
    classe, confianca = clf.classificar(dados.texto)
    return ClassificacaoResposta(
        classe=classe,
        confianca=float(confianca)
    )
