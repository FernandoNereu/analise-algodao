# Projeto de Análise de Dados do Algodão no Brasil 🌾

Este projeto analisa dados históricos de plantio de algodão e variáveis climáticas no Brasil, utilizando ferramentas modernas de ciência de dados para gerar insights sobre tendências, regiões promissoras e correlações entre fatores climáticos e área plantada.

## **Objetivo**

O principal objetivo deste projeto é responder às seguintes questões:

1. **Melhores períodos para plantio:** Quais épocas do ano apresentam condições climáticas ideais para o cultivo de algodão?
2. **Regiões com maior potencial:** Quais estados ou regiões no Brasil oferecem o maior potencial para o plantio de algodão?
3. **Influências climáticas:** Como as variáveis climáticas (temperatura, precipitação, umidade, etc.) impactam a área plantada de algodão?

## **Tecnologias Utilizadas**

- **Linguagem:** Python 3.9+
- **Bibliotecas:**
  - `pandas`, `numpy`: Manipulação e análise de dados.
  - `matplotlib`, `seaborn`, `plotly`: Visualização de dados.
  - `scikit-learn`: Modelagem preditiva.
  - `folium`: Mapas interativos.
  - `streamlit`: Interface de usuário interativa para apresentação do projeto.
- **Containerização:** Docker.

## **Estrutura do Projeto**

``` bash
.
├── data/
│   ├── raw/                 # Dados brutos (históricos e climáticos)
│   ├── processed/           # Dados processados prontos para análise
├── src/
│   ├── app.py               # Aplicação principal Streamlit
│   ├── data_cleaning.py     # Funções de limpeza e pré-processamento
│   ├── analysis.py          # Módulos de análise de dados
│   ├── visualization.py     # Funções de visualização (gráficos e mapas)
├── assets/
│   ├── img/                 # Imagens
├── requirements.txt         # Dependências do projeto
├── Dockerfile               # Arquivo Docker para execução do projeto
└── README.md                # Documentação do projeto
```

## **Instruções para Uso**

- Acesse: https://drive.google.com/drive/folders/15INB0C3GHfH7zBaZHpwy9C73NJF2LhxC?usp=drive_link
  - Para pegar o dataset weather_sum_all.csv (O mesmo é muito grande para o github)

### **Executando Localmente**

1. **Clone o repositório:**

   ```bash
   git clone
   cd analise_algodao
   ```

2. **Crie um ambiente virtual e instale as dependências:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Execute a aplicação:**

   ```bash
   streamlit run src/app.py
   ```

### **Executando com Docker**

1. **Construa a imagem Docker:**

   ```bash
   docker build -t algodao-analise .
   ```

2. **Inicie o container:**

   ```bash
   docker run -p 8501:8501 algodao-analise
   ```

3. **Acesse a aplicação:**  
   Abra o navegador e vá para [http://localhost:8501](http://localhost:8501).

## **Principais Funcionalidades**

1. **Tendências Sazonais:**
   - Identificação de padrões climáticos ao longo dos anos.
   - Gráficos interativos para análise de temperatura, precipitação, e outras variáveis.

2. **Análise Regional:**
   - Mapeamento das melhores regiões para plantio utilizando dados geoespaciais.
   - Visualização interativa de mapas coropléticos.

3. **Influência Climática:**
   - Avaliação das correlações entre variáveis climáticas e área plantada.
   - Gráficos para identificar os fatores climáticos mais influentes.

4. **Tendências Históricas:**
   - Análise de séries temporais para identificar variações na área plantada ao longo das décadas.

5. **Previsão de Área Plantada:**
   - Uso de regressão linear para prever tendências futuras de plantio.

## **Principais Insights**

- **Melhores períodos para plantio:** Primavera e verão destacam-se como os períodos mais favoráveis, devido às temperaturas adequadas e precipitação ideal.
- **Regiões promissoras:** Nordeste e Centro-Oeste apresentam maior potencial devido a fatores climáticos e estruturais.
- **Fatores climáticos relevantes:** Temperatura média e precipitação estão entre os fatores mais fortemente correlacionados com a área plantada.

## **Próximos Passos**

- Incorporar aprendizado de máquina para prever rendimentos com base em variáveis climáticas.
- Expandir os dados para incluir novas regiões e variáveis.
- Implementar análises mais avançadas, como detecção de anomalias e clusterização de regiões.

