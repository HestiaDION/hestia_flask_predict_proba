import os
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging
from flasgger import Swagger

app = Flask(__name__)

# Configuração do Swagger
swagger = Swagger(app)

# Configuração básica do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/predict_user', methods=['POST'])
def predict_user():
    """
    Prever probabilidade positiva de um modelo KNN
    ---
    tags:
      - Previsão de Probabilidade
    parameters:
      - name: body
        in: body
        required: true
        description: JSON com as informações do usuário para previsão
        schema:
          type: object
          required:
            - tipo_de_instituicao
            - idade
            - renda_mensal
            - possui_dne
            - mudanca_de_residencia
            - faculdade_possui_alojamento
            - frequencia_de_uso_de_apps_de_moradia
            - confianca_em_avaliacoes_de_outros_usuarios
          properties:
            tipo_de_instituicao:
              type: string
              description: Tipo de instituição de ensino
            idade:
              type: integer
              description: Idade do usuário
            renda_mensal:
              type: string
              description: Renda mensal do usuário
            possui_dne:
              type: string
              description: Indicação se o usuário possui DNE
            mudanca_de_residencia:
              type: string
              description: Se o usuário já mudou de residência
            faculdade_possui_alojamento:
              type: string
              description: Se a faculdade possui alojamento
            frequencia_de_uso_de_apps_de_moradia:
              type: string
              description: Frequência de uso de apps de moradia
            confianca_em_avaliacoes_de_outros_usuarios:
              type: string
              description: Grau de confiança em avaliações de outros usuários
    responses:
      200:
        description: Sucesso na previsão
        schema:
          type: object
          properties:
            probability:
              type: number
              description: Probabilidade da classe positiva
      400:
        description: Erro de validação
      500:
        description: Erro no servidor
    """
    try:
        # Obtenha os dados JSON enviados no corpo da requisição
        data = request.get_json()

        # Validação básica para garantir que todos os campos estão presentes
        required_fields = [
            "tipo_de_instituicao",
            "idade",
            "renda_mensal",
            "possui_dne",
            "mudanca_de_residencia",
            "faculdade_possui_alojamento",
            "frequencia_de_uso_de_apps_de_moradia",
            "confianca_em_avaliacoes_de_outros_usuarios"
        ]

        for field in required_fields:
            if field not in data:
                logging.warning(f"Campo '{field}' não está presente no JSON")
                return jsonify({"error": f"Campo '{field}' não está presente no JSON"}), 400

        # Criar DataFrame a partir do dicionário recebido
        df = pd.DataFrame([data])

        logging.info(f"Requisição recebida com os seguintes dados: {data}")

        # Carregar o modelo
        try:
            model = joblib.load('model.pkl')
        except FileNotFoundError:
            logging.error("Modelo 'model.pkl' não encontrado")
            return jsonify({"error": "Modelo não encontrado"}), 500
        except Exception as e:
            logging.error(f"Erro ao carregar o modelo: {str(e)}")
            return jsonify({"error": f"Erro ao carregar o modelo: {str(e)}"}), 500

        # Fazer a previsão de probabilidades
        try:
            proba = model.predict_proba(df)
        except Exception as e:
            logging.error(f"Erro ao realizar a previsão: {str(e)}")
            return jsonify({"error": f"Erro ao realizar a previsão: {str(e)}"}), 500

        # Acessar o segundo valor das probabilidades (classe positiva)
        probability = proba[0][1]

        logging.info(f"Previsão realizada com sucesso. Probabilidade positiva: {probability}")

        return jsonify({
            "probability": probability
        }), 200

    except Exception as e:
        logging.error(f"Erro geral no processamento da requisição: {str(e)}")
        return jsonify({"error": "Erro no processamento da requisição"}), 500


if __name__ == '__main__':
    # Defina o host para '0.0.0.0' e a porta a partir da variável de ambiente
    port = int(os.environ.get('PORT', 5000))  # A Render define a porta na variável 'PORT'
    app.run(host='0.0.0.0', port=port, debug=True)
