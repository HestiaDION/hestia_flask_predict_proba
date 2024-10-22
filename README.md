# API Flask com Modelo e Documentação Swagger

Este projeto implementa uma API em Flask que utiliza um modelo de Machine Learning para prever a probabilidade de um resultado positivo com base em dados fornecidos. A API também inclui documentação interativa usando Swagger.

## Funcionalidades

-   **Predição de probabilidade**: A API recebe dados sobre instituições de ensino, idade, renda mensal e outros parâmetros, e retorna a probabilidade de um resultado positivo.
-   **Documentação Swagger**: A API está documentada com Swagger, permitindo que os usuários interajam com ela diretamente a partir de um navegador web.
-   **Modelo**: Utiliza um modelo previamente treinado, carregado a partir de um arquivo `.pkl` serializado.

## Requisitos

-   Python 3.7+
-   Pip (para instalação de pacotes)

## Instalação

1.  **Clone o repositório**:
    ```
	git clone https://github.com/HestiaDION/hestia_flask_predict_proba
	```
    
2.  **Crie e ative um ambiente virtual** (opcional, mas recomendado):
    ```
    python3 -m venv venv
    
    source venv/bin/activate  # No Windows, use venv\Scripts\activate
    ```
3.  **Instale as dependências**:
    
    Instale as dependências a partir do arquivo `requirements.txt`:
    
    bash
    
    Copiar código
    
    `pip install -r requirements.txt` 
    

## Como Rodar a API

1.  **Inicie a API**:
    
    Execute o arquivo `app.py` para iniciar o servidor Flask:

    `python app.py` 
    
    Isso iniciará a API Flask na porta 5000 por padrão.
    
2.  **Acesse a API**:
    
    A API estará disponível em: `http://127.0.0.1:5000` 
    

## Testando a API com Swagger

1.  **Documentação Interativa**:
    
    Após iniciar o servidor, você pode acessar a documentação Swagger em: `http://127.0.0.1:5000/apidocs/` 
    
    Na página Swagger, você poderá testar os endpoints da API diretamente pelo navegador.
    

## Endpoints

### POST `/post-data`

Recebe um JSON com os dados do usuário e retorna a probabilidade de um resultado positivo.

-   **URL**: `/post-data`
    
-   **Método**: `POST`
    
-   **Parâmetros no corpo**:
    
    -   `Tipo de Instituição` (string): Tipo de instituição (ex.: "Pública", "Privada").
    
    -   `Idade` (int): Idade do usuário.
    -   `Renda Mensal` (string): Renda mensal do usuário (ex.: "Não possuo renda", "3500").
    -   `Possui DNE` (string): Se o usuário possui DNE (ex.: "Sim", "Não").
    -   `Mudança de Residência` (string): Se o usuário já mudou de residência (ex.: "Sim", "Não").
    -   `Faculdade Possui Alojamento` (string): Se a faculdade oferece alojamento (ex.: "Sim", "Não").
    -   `Frequência de Uso de Apps de Moradia` (string): Frequência de uso de aplicativos de moradia (ex.: "Diário", "Semanal").
    -   `Confiança em Avaliações de Outros Usuários` (string): Grau de confiança em avaliações de outros usuários (ex.: "Alta", "Baixa").
-   **Exemplo de JSON de entrada**:
    ```
    
    {
      "Tipo de Instituição": "Pública",
      "Idade": 37,
      "Renda Mensal": "Não possuo renda",
      "Possui DNE": "Não",
      "Mudança de Residência": "Não",
      "Faculdade Possui Alojamento": "Não",
      "Frequência de Uso de Apps de Moradia": "Baixa",
      "Confiança em Avaliações de Outros Usuários": "Baixa"
    }
    ```
    
-   **Exemplo de resposta**:
   
    ```
    {
      "probability": 0.65
    }
    ```

## Estrutura do Projeto

```
/hestia_flask_predict_proba
├── app.py                    # Código principal da API
├── model.pkl  				  # Arquivo do modelo serializado
├── requirements.txt          # Dependências do projeto
└── README.md                 # Documentação do projeto
```

## Dependências

As dependências do projeto estão listadas no arquivo `requirements.txt`.