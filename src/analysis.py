import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import streamlit as st
from deap import base, creator, tools, algorithms


def analyze_seasonal_trends(
    cotton_data: pd.DataFrame, weather_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Analisa tendências sazonais combinando dados de algodão e climáticos.
    """
    try:
        # Identificar colunas numéricas
        numeric_cols = weather_data.select_dtypes(include=[float, int]).columns.tolist()

        # Agrupar os dados climáticos por ano e estação, calculando a média
        seasonal_weather = weather_data.groupby(["Ano", "Estacao"], as_index=False)[
            numeric_cols
        ].mean()

        # Combinar dados de algodão com as tendências sazonais climáticas
        combined_data = pd.merge(cotton_data, seasonal_weather, on="Ano", how="inner")

        #print("Pré-visualização dos dados sazonais combinados:")
        #print(combined_data.head())

        return combined_data
    except Exception as e:
        raise RuntimeError(f"Erro ao analisar tendências sazonais: {e}")


def analyze_regional_potential(cotton_data, weather_data):
    """
    Analisa as melhores regiões para o plantio de algodão.
    """
    try:
        # Inspecionar e garantir que todas as colunas numéricas sejam numéricas
        numeric_cols = ["Area_Plantada"]  # Atualize conforme necessário
        for col in numeric_cols:
            cotton_data[col] = pd.to_numeric(cotton_data[col], errors="coerce")

        # Remover linhas com valores NaN nas colunas numéricas
        cotton_data = cotton_data.dropna(subset=numeric_cols)

        # Garantir que os dados estejam prontos para agrupamento
        #print("Pré-visualização dos dados antes do agrupamento:")
        #print(cotton_data.head())

        # Agrupar por região e calcular a média da área plantada
        regional_data = (
            cotton_data.groupby("Região/UF")[numeric_cols]
            .mean()
            .sort_values(by="Area_Plantada", ascending=False)
        )

        # Resetar o índice para facilitar a visualização
        regional_data = regional_data.reset_index()

        #print("Pré-visualização das melhores regiões para plantio:")
        #print(regional_data.head())

        return regional_data
    except Exception as e:
        raise RuntimeError(f"Erro ao analisar potencial regional: {e}")


def analyze_climatic_influences(cotton_data, weather_data):
    # Garantir que 'Região/UF' exista em ambos os datasets
    if "Região/UF" not in weather_data.columns:
        # Exemplo de mapeamento; ajuste conforme necessário
        station_to_region = {
            "A001": "NORTE",
            "A002": "NORDESTE",
            # Outros mapeamentos
        }
        weather_data["Região/UF"] = weather_data["ESTACAO"].map(station_to_region)

    # Certificar-se de que a coluna 'Ano' existe e está correta
    if "Ano" not in weather_data.columns:
        weather_data["Ano"] = pd.to_datetime(weather_data["DATA (YYYY-MM-DD)"]).dt.year

    # Realizar a mesclagem
    combined_data = cotton_data.merge(
        weather_data, on=["Ano", "Região/UF"], how="inner"
    )

    # Filtrar apenas colunas numéricas
    numeric_data = combined_data.select_dtypes(include=["float64", "int64"])

    # Calcular correlações
    correlations = numeric_data.corr()["Area_Plantada"].sort_values(ascending=False)

    return correlations


def analyze_historical_trends(cotton_data):
    # Garantir que o nome da coluna esteja correto
    if "Area_Planted" not in cotton_data.columns:
        cotton_data.rename(columns={"Area_Plantada": "Area_Planted"}, inplace=True)

    # Agrupar por ano e somar a área plantada
    historical_trends = cotton_data.groupby("Ano")["Area_Planted"].sum().reset_index()

    # Criar visualização
    plt.figure(figsize=(10, 6))
    plt.plot(historical_trends["Ano"], historical_trends["Area_Planted"], marker="o")
    plt.title("Tendências Históricas de Plantio de Algodão")
    plt.xlabel("Ano")
    plt.ylabel("Área Plantada (em mil hectares)")
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    return historical_trends


def predict_planted_area(cotton_data, years_to_consider=10, forecast_until=2030):
    try:
        cotton_data = cotton_data.rename(columns={"Area_Plantada": "Area_Planted"})
        recent_years = sorted(cotton_data["Ano"].unique())[-years_to_consider:]
        filtered_data = cotton_data[cotton_data["Ano"].isin(recent_years)].copy()

        # Validar dados
        if filtered_data.empty or filtered_data["Area_Planted"].isnull().all():
            raise ValueError("Dados insuficientes para previsão.")

        X = filtered_data["Ano"].values.reshape(-1, 1)
        y = filtered_data["Area_Planted"].values

        # Regressão polinomial
        poly = PolynomialFeatures(degree=2)  # Ajuste o grau conforme necessário
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)

        # Prever anos futuros
        future_years = np.arange(X[-1][0] + 1, forecast_until + 1).reshape(-1, 1)
        future_predictions = model.predict(poly.transform(future_years))

        # Criar DataFrame de previsões
        predictions = pd.DataFrame(
            {
                "Ano": future_years.flatten(),
                "Area_Planted_Predicted": future_predictions,
            }
        )

        return predictions
    except Exception as e:
        raise RuntimeError(f"Erro ao prever área plantada: {e}")


def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Pré-processa os dados de área plantada de algodão.
    """
    try:
        # Carregar os dados
        data = pd.read_csv(file_path)

        # Transformar colunas de formato largo para formato longo
        data_long = data.melt(
            id_vars=["REGIÃO/UF"], var_name="Ano", value_name="Area_Plantada"
        )

        # Remover separadores de milhar e transformar decimais
        data_long["Area_Plantada"] = (
            data_long["Area_Plantada"]
            .str.replace(",", ".", regex=False)
            .str.replace(".", "", regex=False)
            .astype(float)
        )

        # Converter coluna 'Ano' para numérico
        data_long["Ano"] = pd.to_numeric(
            data_long["Ano"].str.extract(r"(\d{4})")[0], errors="coerce"
        )

        # Remover valores ausentes
        data_long = data_long.dropna(subset=["Ano", "Area_Plantada"])

        return data_long
    except Exception as e:
        raise RuntimeError(f"Erro ao pré-processar dados: {e}")

def monte_carlo_simulation(data, num_simulations=1000, forecast_years=10):
    """
    Realiza simulações de Monte Carlo para prever a área plantada de algodão.
    
    Args:
        data (pd.DataFrame): Dados históricos de área plantada.
        num_simulations (int): Número de simulações a serem realizadas.
        forecast_years (int): Número de anos a serem previstos.
    
    Returns:
        pd.DataFrame: Resultados das simulações.
    """
    # Extrair os dados históricos
    historical_data = data["Area_Planted"].values
    last_year = data["Ano"].max()
    
    # Calcular a média e o desvio padrão dos dados históricos
    mean = np.mean(historical_data)
    std_dev = np.std(historical_data)
    
    # Realizar as simulações
    simulations = []
    for _ in range(num_simulations):
        forecast = [mean + std_dev * np.random.randn() for _ in range(forecast_years)]
        simulations.append(forecast)
    
    # Criar um DataFrame com os resultados
    simulation_df = pd.DataFrame(simulations, columns=[f"Ano_{last_year + i + 1}" for i in range(forecast_years)])
    
    return simulation_df

#v2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def advanced_prediction(cotton_data, weather_data):
    """
    Realiza previsões usando um modelo de Random Forest.
    """
    try:
        # Combinar os dados
        combined_data = pd.merge(cotton_data, weather_data, on=["Ano", "Região/UF"], how="inner")
        
        # Selecionar features e target
        features = combined_data[["temp_avg", "rain_max", "hum_max", "wind_avg"]]
        target = combined_data["Area_Planted"]
        
        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Treinar o modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Fazer previsões
        y_pred = model.predict(X_test)
        
        # Calcular métricas de avaliação
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Adicionar previsões ao DataFrame
        combined_data["Area_Planted_Predicted"] = model.predict(features)
        
        # Retornar resultados e métricas
        return combined_data, mae, r2
    except Exception as e:
        raise RuntimeError(f"Erro ao realizar previsão avançada: {e}")