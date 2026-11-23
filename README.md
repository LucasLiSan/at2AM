# EduPredict — Classificação de Desempenho Acadêmico com Redes Neurais MLP

Projeto acadêmico desenvolvido para a atividade avaliativa **AT2 — Redes Neurais MLP**, da disciplina de Aprendizagem de Máquina, no curso de Desenvolvimento de Software Multiplataforma.

O EduPredict utiliza técnicas de aprendizado de máquina para analisar fatores associados ao desempenho acadêmico e classificar estudantes em três faixas relativas de desempenho: **BAIXO, MEDIO e ALTO**.

A solução integra análise exploratória em Jupyter Notebook, pré-processamento de dados, treinamento de uma rede neural Multilayer Perceptron (MLP) e disponibilização das previsões por meio de uma API REST desenvolvida com FastAPI.

## 1. Tecnologias utilizadas

- **Python** — linguagem de programação.
- **UV** — gerenciamento de dependências e ambiente virtual.
- **Jupyter Notebook** — análise exploratória e treinamento.
- **Pandas e NumPy** — manipulação e análise dos dados.
- **Matplotlib e Seaborn** — visualização de dados.
- **Scikit-learn** — pré-processamento, treinamento e avaliação da MLP.
- **Joblib** — persistência do modelo treinado.
- **FastAPI** — desenvolvimento da API REST.
- **Pydantic** — validação dos dados recebidos.

## 2. Dataset

Foi utilizado o dataset público **Student Performance Factors**, disponibilizado no Kaggle.

Fonte: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

O conjunto de dados contém:

- **6.607 registros**;
- **20 colunas**;
- **19 atributos preditores**;
- **1 variável alvo original:** `Exam_Score`;
- **0 registros duplicados**.

Entre os atributos analisados estão frequência escolar, horas de estudo, notas anteriores, motivação, participação familiar, acesso a recursos educacionais, qualidade do ensino e hábitos de estudo.

A variável `Exam_Score` representa a nota do exame. A partir dela, foram criadas três classes relativas de desempenho.

## 3. Definição das classes

As classes foram definidas utilizando os percentis de 33,3% e 66,7% calculados exclusivamente sobre as notas do conjunto de treinamento.

| Classe | Critério |
|---|---|
| BAIXO | Nota menor ou igual a 66 |
| MEDIO | Nota maior que 66 e menor ou igual a 69 |
| ALTO | Nota maior que 69 |

Essas categorias representam faixas estatísticas do dataset e não correspondem a critérios oficiais de aprovação ou reprovação escolar.

## 4. Pré-processamento

O pré-processamento foi desenvolvido utilizando `ColumnTransformer` e `Pipeline`, do Scikit-learn.

Para os atributos numéricos, foram utilizados:

- `SimpleImputer(strategy="median")`, para tratamento de valores ausentes;
- `StandardScaler`, para padronização dos valores.

Para os atributos categóricos:

- `SimpleImputer(strategy="most_frequent")`;
- `OneHotEncoder(handle_unknown="ignore")`.

O pré-processamento foi integrado ao pipeline da rede neural, permitindo reutilizar as mesmas transformações durante as previsões da API.

## 5. Arquitetura da rede neural

O modelo utiliza `MLPClassifier`, com a seguinte configuração:

| Parâmetro | Configuração |
|---|---|
| Camadas ocultas | 2 |
| Neurônios | 64 e 32 |
| Função de ativação | ReLU |
| Otimizador | Adam |
| Taxa de aprendizado inicial | 0.001 |
| Máximo de iterações | 500 |
| Early stopping | Ativado |
| Random state | 42 |

O dataset foi dividido em **80% para treinamento e 20% para teste**.

## 6. Resultados obtidos

O modelo apresentou os seguintes resultados no conjunto de teste:

**Acurácia: 96,44%**

| Classe | Precisão | Recall | F1-score | Amostras |
|---|---:|---:|---:|---:|
| BAIXO | 0.96 | 0.98 | 0.97 | 567 |
| MEDIO | 0.95 | 0.94 | 0.95 | 407 |
| ALTO | 0.99 | 0.96 | 0.97 | 348 |

O F1-score macro foi de aproximadamente **0,96**.

O desempenho foi avaliado por meio de métricas de classificação e matriz de confusão. Esses resultados se referem exclusivamente ao conjunto de dados utilizado no experimento e não garantem o mesmo desempenho em ambientes escolares reais.

## 7. Estrutura do projeto

```text
at2AM/
├── data/
│   └── dataset.csv
├── models/
│   └── modelo_mlp.joblib
├── notebooks/
│   └── analise_treinamento.ipynb
├── src/
│   ├── __init__.py
│   ├── app.py
│   └── schemas.py
├── requisicoes.http
├── pyproject.toml
├── uv.lock
└── README.md
```

O arquivo `modelo_mlp.joblib` contém o pipeline treinado e os metadados necessários à aplicação.

## 8. Instalação e execução

### Pré-requisitos

- Git
- UV
- Python 3.12, gerenciado pelo UV

### Clonar o repositório

```bash
git clone https://github.com/LucasLiSan/at2AM.git
cd at2AM
```

### Instalar as dependências

```bash
uv sync
```

### Executar o Jupyter Notebook

```bash
uv run jupyter lab
```

Abra o arquivo `notebooks/analise_treinamento.ipynb` para consultar a análise exploratória, o treinamento e a avaliação da rede neural.

Caso o modelo ainda não esteja disponível, execute as células de treinamento e exportação do Notebook para gerar `models/modelo_mlp.joblib`.

### Iniciar a API REST

```bash
uv run uvicorn src.app:app --reload
```

A aplicação estará disponível em:

http://127.0.0.1:8000

Documentação interativa Swagger:

http://127.0.0.1:8000/docs

## 9. Endpoints da API

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/` | Informações iniciais |
| GET | `/health` | Verifica o funcionamento da API |
| GET | `/model-info` | Retorna informações do modelo |
| POST | `/predict` | Realiza uma classificação |

### Exemplo de requisição

**POST `/predict`**

```json
{
  "Hours_Studied": 20,
  "Attendance": 85,
  "Parental_Involvement": "Medium",
  "Access_to_Resources": "High",
  "Extracurricular_Activities": "Yes",
  "Sleep_Hours": 7,
  "Previous_Scores": 75,
  "Motivation_Level": "Medium",
  "Internet_Access": "Yes",
  "Tutoring_Sessions": 2,
  "Family_Income": "Medium",
  "Teacher_Quality": "Good",
  "School_Type": "Public",
  "Peer_Influence": "Positive",
  "Physical_Activity": 3,
  "Learning_Disabilities": "No",
  "Parental_Education_Level": "College",
  "Distance_from_Home": "Near",
  "Gender": "Female"
}
```

### Estrutura da resposta

A API retorna um JSON contendo:

- `classificacao`: classe prevista pela MLP;
- `confianca`: probabilidade atribuída à classe prevista;
- `probabilidades`: probabilidades das três classes;
- `mensagem`: descrição da faixa prevista.

Os valores são calculados pelo modelo treinado durante a execução.

## 10. Validação de dados

A API utiliza Pydantic para validar as entradas.

São verificadas condições como:

- Campos obrigatórios;
- Tipos de dados;
- Limites numéricos;
- Categorias permitidas;
- Rejeição de atributos desconhecidos.

Requisições inválidas recebem resposta HTTP `422 Unprocessable Entity`.

O arquivo `requisicoes.http` contém exemplos para testar os endpoints.

## 11. Considerações éticas e limitações

O EduPredict foi desenvolvido para fins acadêmicos e demonstra o uso de redes neurais artificiais em um contexto educacional.

As classificações representam padrões aprendidos a partir de um dataset público e não devem ser utilizadas isoladamente para avaliar estudantes, definir aprovações, aplicar sanções ou tomar decisões pedagógicas.

O desempenho observado pode não se repetir em dados reais de outras instituições. A utilização prática exigiria validação adicional, análise de possíveis vieses, proteção de dados pessoais e supervisão humana.

## 12. Repositório

Código-fonte disponível em:

https://github.com/LucasLiSan/at2AM

---

**Projeto acadêmico — Aprendizagem de Máquina — AT2 Redes Neurais MLP.**