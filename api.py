from flask import Flask, request, jsonify
from main import ClassificadorPerguntasBERT

app = Flask(__name__)

clf = ClassificadorPerguntasBERT()
clf.carregar_modelo("modelo")

@app.route("/classificar", methods=["POST"])
def classificar():
    data = request.get_json()

    if "texto" not in data:
        return jsonify({"erro": "campo 'texto' ausente"}), 400

    classe, confianca = clf.classificar(data["texto"])

    return jsonify({
        "classe": classe,
        "confianca": confianca
    })

if __name__ == "__main__":
    app.run(debug=True)
