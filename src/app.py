import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from data_cleaning import load_cotton_data, load_weather_data
from analysis import (
    analyze_seasonal_trends,
    analyze_regional_potential,
    analyze_climatic_influences,
    analyze_historical_trends,
    predict_planted_area,
    monte_carlo_simulation,
    advanced_prediction,
)
from visualization import (
    plot_seasonal_trends,
    plot_regional_map,
    plot_climatic_influence,
    plot_historical_trends,
    plot_correlation_heatmap,
    plot_historical_trends_with_prediction,
    plot_interactive_line,
    add_coordinates_to_regions,
)

# Diretório base ajustado
#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
#GEO_DIR = os.path.join(BASE_DIR, "data", "geo")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = BASE_DIR
GEO_DIR = BASE_DIR


# Configuração inicial da página
st.set_page_config(page_title="Análise de Algodão no Brasil", layout="wide")

# Título e introdução
st.title("Análise de Dados de Plantio e Colheita de Algodão no Brasil")
st.markdown(
    """
    Este painel interativo oferece insights sobre dados históricos de algodão e condições climáticas no Brasil. 
    Descubra os melhores períodos para plantio, regiões promissoras, tendências históricas e muito mais.
    """
)

# Carregar dados
st.sidebar.header("Carregar Dados")
try:
    #cotton_data_path = os.path.join(DATA_DIR, "AlgodoSerieHist.xlsx")
    #weather_data_path = os.path.join(DATA_DIR, "weather_sum_all.csv")
    cotton_data_path = os.path.join(DATA_DIR, "AlgodoSerieHist.xlsx")
    weather_data_path = os.path.join(DATA_DIR, "weather_sum_all.csv")

    cotton_data = load_cotton_data(cotton_data_path)
    weather_data = load_weather_data(weather_data_path)

    st.sidebar.success("Dados carregados com sucesso!")
except Exception as e:
    st.sidebar.error(f"Erro ao carregar dados: {e}")
    st.stop()

# Sidebar para exibir dados brutos
if st.sidebar.checkbox("Exibir dados brutos de algodão"):
    st.subheader("Dados Brutos de Algodão")
    st.write(cotton_data.head(20))

if st.sidebar.checkbox("Exibir dados meteorológicos brutos"):
    st.subheader("Dados Brutos Meteorológicos")
    st.write(weather_data.head(20))

# Tabs principais
tabs = st.tabs(
    [
        "Tendências Sazonais",
        "Melhores Regiões",
        "Influência Climática",
        "Tendências Históricas",
        "Correlação de Variáveis",
        "Previsão de Area Plantada",
        "Conclusões",        
        "Modelo Avançado (Random Forest)",
    ]
)

# Aba: Tendências Sazonais
with tabs[0]:
    st.header("Tendências Sazonais")
    try:
        seasonal_trends = analyze_seasonal_trends(cotton_data, weather_data)
        st.subheader("Gráfico")
        plot_seasonal_trends(seasonal_trends)
        st.subheader("Dados de Tendências Sazonais")
        st.write(seasonal_trends)
    except Exception as e:
        st.error(f"Erro ao analisar tendências sazonais: {e}")

# Aba: Melhores Regiões
with tabs[1]:
    st.header("Melhores Regiões para Plantio")
    try:
        regional_potential = analyze_regional_potential(cotton_data, weather_data)
        st.subheader("Mapa")

        # Adicione o caminho correto para o shapefile
        shapefile_path = "./data/geo/br_states.json"
        plot_regional_map(regional_potential, shapefile_path)

        st.subheader("Detalhes por Região")
        st.write(regional_potential)
    except Exception as e:
        st.error(f"Erro ao analisar regiões: {e}")

# Aba: Influência Climática
with tabs[2]:
    st.header("Influência Climática")
    try:
        climatic_influences = analyze_climatic_influences(cotton_data, weather_data)
        st.subheader("Gráfico")
        plot_climatic_influence(climatic_influences)
        st.subheader("Detalhes da Influência Climática")
        st.write(climatic_influences)
    except Exception as e:
        st.error(f"Erro ao analisar influências climáticas: {e}")

# Aba: Tendências Históricas
with tabs[3]:
    st.header("Tendências Históricas")
    try:
        historical_trends = analyze_historical_trends(cotton_data)
        plot_interactive_line(historical_trends, "Ano", "Area_Planted", 
                             "Tendências Históricas da Área Plantada", "Ano", "Área Plantada (ha)")
    except Exception as e:
        st.error(f"Erro ao analisar tendências históricas: {e}")

# Aba: Correlação de Variáveis
with tabs[4]:
    st.header("Mapa de Correlação")
    try:
        st.subheader("Mapa de Calor")
        plot_correlation_heatmap(cotton_data, weather_data)
    except Exception as e:
        st.error(f"Erro ao gerar mapa de correlação: {e}")


# Aba: Previsão
with tabs[5]:
    st.header("Previsão da Área Plantada")

    try:
        # Entrada para selecionar o número de anos a considerar
        years_to_consider = st.number_input(
            "Anos para considerar na previsão:",
            min_value=2,
            max_value=len(cotton_data["Ano"].unique()),
            value=10,
            step=1,
        )
        # Análise de tendências históricas

        if historical_trends.empty:
            st.error("Dados históricos de área plantada não estão disponíveis.")
        else:
            # Limpar e validar dados históricos
            historical_trends["Ano"] = pd.to_numeric(
                historical_trends["Ano"], errors="coerce"
            )
            historical_trends["Area_Planted"] = pd.to_numeric(
                historical_trends["Area_Planted"], errors="coerce"
            )
            historical_trends = historical_trends.dropna(subset=["Ano", "Area_Planted"])

            # Filtrar os anos recentes
            recent_years = sorted(historical_trends["Ano"].unique())[
                -years_to_consider:
            ]
            filtered_historical_trends = historical_trends[
                historical_trends["Ano"].isin(recent_years)
            ]
            st.write("Dados Históricos Filtrados:", filtered_historical_trends)

            st.subheader("Previsão de Área Plantada")

            if len(filtered_historical_trends) < 2:
                st.warning(
                    "Dados insuficientes para previsão. É necessário pelo menos dois anos de dados históricos."
                )
            else:
                try:
                    # Previsão com dados filtrados
                    predicted_areas = predict_planted_area(
                        filtered_historical_trends, years_to_consider=years_to_consider
                    )

                    if predicted_areas.empty:
                        st.warning(
                            "Não foi possível gerar previsões para a área plantada."
                        )
                    else:
                        st.write("Previsão de Área Plantada (ha):")
                        st.write(predicted_areas)

                        # Gráfico com histórico e previsão
                        plot_historical_trends_with_prediction(
                            filtered_historical_trends, predicted_areas
                        )

                        st.success("Análise e previsão concluídas com sucesso!")
                except RuntimeError as prediction_error:
                    st.error(f"Erro ao prever área plantada: {prediction_error}")

    except Exception as e:
        st.error(f"Erro ao analisar tendências históricas com previsão: {e}")


with tabs[6]:
    st.header("Simulações de Monte Carlo")
    st.markdown(
        """
        As simulações de Monte Carlo são usadas para prever a área plantada de algodão nos próximos anos. 
        Elas geram várias previsões possíveis com base em dados históricos, permitindo entender a variação e a incerteza nas previsões.
        """
    )
    try:
        num_simulations = st.number_input("Número de Simulações:", min_value=100, max_value=10000, value=1000, step=100, key="num_simulations_mc")
        forecast_years = st.number_input("Anos de Previsão:", min_value=1, max_value=20, value=10, step=1, key="forecast_years_mc")
        
        # Realizar as simulações de Monte Carlo
        simulation_results = monte_carlo_simulation(historical_trends, num_simulations, forecast_years)
        
        st.subheader("Distribuição das Simulações")
        st.markdown(
            """
            O gráfico abaixo mostra a distribuição das áreas plantadas previstas pelas simulações de Monte Carlo. 
            A linha vermelha representa a média das previsões, enquanto a área sombreada mostra a variação esperada.
            """
        )
        plt.figure(figsize=(10, 6))
        sns.histplot(simulation_results.values.flatten(), bins=50, kde=True, color='skyblue', label="Distribuição")
        plt.axvline(simulation_results.values.flatten().mean(), color='red', linestyle='--', label="Média")
        plt.title("Distribuição das Simulações de Monte Carlo", fontsize=16)
        plt.xlabel("Área Plantada (ha)", fontsize=14)
        plt.ylabel("Número de Ocorrências", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(plt)

        st.subheader("Relação entre Variáveis Climáticas e Área Plantada")
        st.markdown(
            """
            Os gráficos de dispersão abaixo mostram a relação entre as variáveis climáticas e a área plantada prevista.
            """
        )
        
        # Definir distribuições de probabilidade para variáveis climáticas
        temperatura_media = np.random.normal(loc=25, scale=5, size=num_simulations)  # média de 25°C, desvio padrão de 5
        precipitacao = np.random.normal(loc=100, scale=30, size=num_simulations)  # média de 100mm, desvio padrão de 30mm
        area_plantada = simulation_results.values.flatten()[:num_simulations]  # Garantir o mesmo comprimento

        # Criar DataFrame com os resultados das simulações
        climatic_simulations = pd.DataFrame({
            "Temperatura Média (°C)": temperatura_media,
            "Precipitação (mm)": precipitacao,
            "Área Plantada (ha)": area_plantada
        })

        # Gráfico de Dispersão: Temperatura vs Área Plantada
        st.markdown("#### Relação entre Temperatura e Área Plantada")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='Temperatura Média (°C)', 
            y='Área Plantada (ha)', 
            data=climatic_simulations, 
            hue='Precipitação (mm)',  # Cor dos pontos baseada na precipitação
            palette='coolwarm',  # Escala de cores
            size='Precipitação (mm)',  # Tamanho dos pontos baseado na precipitação
            sizes=(50, 200),  # Tamanho mínimo e máximo dos pontos
            alpha=0.7,  # Transparência dos pontos
            ax=ax
        )
        plt.title("Temperatura Média vs Área Plantada", fontsize=16)
        plt.xlabel("Temperatura Média (°C)", fontsize=14)
        plt.ylabel("Área Plantada (ha)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title="Precipitação (mm)", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

        # Gráfico de Dispersão: Precipitação vs Área Plantada
        st.markdown("#### Relação entre Precipitação e Área Plantada")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='Precipitação (mm)', 
            y='Área Plantada (ha)', 
            data=climatic_simulations, 
            hue='Temperatura Média (°C)',  # Cor dos pontos baseada na temperatura
            palette='viridis',  # Escala de cores
            size='Temperatura Média (°C)',  # Tamanho dos pontos baseado na temperatura
            sizes=(50, 200),  # Tamanho mínimo e máximo dos pontos
            alpha=0.7,  # Transparência dos pontos
            ax=ax
        )
        plt.title("Precipitação vs Área Plantada", fontsize=16)
        plt.xlabel("Precipitação (mm)", fontsize=14)
        plt.ylabel("Área Plantada (ha)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title="Temperatura Média (°C)", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao realizar simulações de Monte Carlo: {e}")

#v2 Aba: Modelo Avançado (Random Forest)
with tabs[7]:
    st.header("Previsão com Modelo Avançado (Random Forest)")
    try:
        # Realizar previsões com o modelo avançado
        advanced_results, mae, r2 = advanced_prediction(cotton_data, weather_data)
        
        # Exibir métricas de avaliação
        st.subheader("Métricas de Avaliação do Modelo")
        st.write(f"**Erro Médio Absoluto (MAE):** {mae:.2f}")
        st.write(f"**Coeficiente de Determinação (R²):** {r2:.2f}")
        
        # Exibir previsões
        st.subheader("Previsões do Modelo")
        st.write(advanced_results)    

        # Gráfico de barras comparando real vs. previsto
        st.subheader("Comparação entre Valores Reais e Previstos")
        fig = px.bar(advanced_results, x="Ano", y=["Area_Planted", "Area_Planted_Predicted"], 
                    barmode="group", title="Comparação entre Valores Reais e Previstos",
                    labels={"value": "Área Plantada (ha)", "variable": "Tipo"})
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Erro ao executar o modelo avançado: {e}")
