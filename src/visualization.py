import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.express as px
import folium
from streamlit_folium import st_folium

@st.cache_data
def prepare_combined_data(cotton_data, weather_data):
    """
    Prepara os dados combinados para análise (merge de algodão e clima).
    """
    # Renomear colunas
    cotton_data.rename(columns={"Area_Plantada": "Area_Planted"}, inplace=True)

    # Filtrar dados por anos em comum
    common_years = set(cotton_data["Ano"]).intersection(weather_data["Ano"])
    cotton_filtered = cotton_data[cotton_data["Ano"].isin(common_years)]
    weather_filtered = weather_data[weather_data["Ano"].isin(common_years)]

    # Fazer o merge dos dados
    combined_data = pd.merge(cotton_filtered, weather_filtered, on="Ano")
    return combined_data


def plot_seasonal_trends(seasonal_data: pd.DataFrame):
    """
    Plota tendências sazonais.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=seasonal_data, x="Ano", y="temp_avg", hue="Estacao")
    plt.title("Tendências Sazonais de Temperatura Média")
    plt.xlabel("Ano")
    plt.ylabel("Temperatura Média (°C)")
    st.pyplot(plt)


def plot_regional_map(regional_data, geojson_path):
    """
    Plota o mapa das melhores regiões para plantio de algodão, focado no Brasil.
    """
    try:
        # Renomear colunas no regional_data para corresponder ao GeoJSON
        if "Região/UF" in regional_data.columns:
            regional_data = regional_data.rename(columns={"Região/UF": "id"})

        # Verificar as colunas após o rename
        #print("Colunas no regional_data após ajuste:", regional_data.columns)

        # Carregar o GeoJSON
        brazil_geo = gpd.read_file(geojson_path)
        #print("Colunas no GeoJSON:", brazil_geo.columns)

        # Verificar se as chaves do merge correspondem
        #print("IDs no GeoJSON:", brazil_geo["id"].unique())
        #print("IDs no regional_data:", regional_data["id"].unique())

        # Criar o mapa centrado no Brasil
        m = folium.Map(location=[-14.235, -51.9253], zoom_start=4)

        # Adicionar o mapa coroplético
        folium.Choropleth(
            geo_data=geojson_path,
            name="choropleth",
            data=regional_data,
            columns=["id", "Area_Plantada"],  # Usar a coluna 'id' e 'Area_Plantada'
            key_on="feature.id",  # Ajustar para usar o campo 'id' do GeoJSON
            fill_color="YlGn",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Área Plantada (ha)",
        ).add_to(m)

        # Adicionar controle de camadas
        folium.LayerControl().add_to(m)

        # Exibir o mapa no Streamlit
        st_folium(m, width=800, height=600)

    except Exception as e:
        st.error(f"Erro ao plotar o mapa interativo: {e}")


def add_coordinates_to_regions(regional_data):
    """
    Adiciona coordenadas de longitude e latitude às regiões no DataFrame.
    """
    # Exemplo de mapeamento de regiões para coordenadas
    coordinates = {
        "NORDESTE": {"lon": -38.0, "lat": -7.0},
        "CENTRO-SUL": {"lon": -47.0, "lat": -21.0},
        "CENTRO-OESTE": {"lon": -56.0, "lat": -16.0},
        "MT": {"lon": -55.0, "lat": -12.0},
        "CE": {"lon": -39.0, "lat": -5.0},
        "BA": {"lon": -41.0, "lat": -12.0},
        "SUDESTE": {"lon": -44.0, "lat": -22.0},
        "PR": {"lon": -51.0, "lat": -24.0},
        "SUL": {"lon": -49.0, "lat": -27.0},
        "PB": {"lon": -36.5, "lat": -7.2},
        "SP": {"lon": -46.6, "lat": -23.5},
        "RN": {"lon": -36.5, "lat": -5.8},
        "PI": {"lon": -42.8, "lat": -6.6},
        "MG": {"lon": -44.0, "lat": -18.0},
        "GO": {"lon": -49.0, "lat": -15.9},
        "PE": {"lon": -37.0, "lat": -8.0},
        "MS": {"lon": -54.0, "lat": -20.4},
        "AL": {"lon": -36.8, "lat": -9.6},
        "MA": {"lon": -44.2, "lat": -5.4},
        "NORTE": {"lon": -60.0, "lat": -3.0},
        "SE": {"lon": -37.0, "lat": -10.6},
        "RO": {"lon": -62.0, "lat": -11.2},
        "PA": {"lon": -52.0, "lat": -1.4},
        "TO": {"lon": -48.0, "lat": -10.2},
        "DF": {"lon": -47.9, "lat": -15.8},
        "RR": {"lon": -61.4, "lat": 2.8},
        "RJ": {"lon": -43.2, "lat": -22.9},
        "RS": {"lon": -51.2, "lat": -30.0},
        "SC": {"lon": -49.0, "lat": -27.6},
        "ES": {"lon": -40.3, "lat": -19.2},
        "AP": {"lon": -51.0, "lat": 0.0},
        "AM": {"lon": -62.0, "lat": -3.0},
        "AC": {"lon": -70.0, "lat": -9.5},
    }

    # Adicionar colunas de longitude e latitude
    regional_data["lon"] = regional_data["Região/UF"].map(
        lambda x: coordinates.get(x, {}).get("lon")
    )
    regional_data["lat"] = regional_data["Região/UF"].map(
        lambda x: coordinates.get(x, {}).get("lat")
    )

    # Verificar se há valores ausentes
    missing = regional_data[
        regional_data["lon"].isnull() | regional_data["lat"].isnull()
    ]
    if not missing.empty:
        #print(f"Coordenadas ausentes para: {missing['Região/UF'].tolist()}")
        raise ValueError("Adicione coordenadas para todas as regiões.")

    return regional_data


def plot_correlation_heatmap(cotton_data, weather_data):
    """
    Plota um mapa de calor de correlação com melhorias de nomeclatura e design.
    """
    try:
        # Combinar os dados
        combined_data = cotton_data.merge(
            weather_data, on=["Ano", "Região/UF"], how="inner"
        )

        # Selecionar apenas colunas numéricas
        numeric_data = combined_data.select_dtypes(include=["float64", "int64"])

        # Calcular a matriz de correlação
        corr_matrix = numeric_data.corr()

        # Renomear variáveis para maior clareza
        rename_dict = {
            "temp_max": "Temperatura Máxima (°C)",
            "temp_avg": "Temperatura Média (°C)",
            "temp_min": "Temperatura Mínima (°C)",
            "hum_max": "Umidade Máxima (%)",
            "hum_min": "Umidade Mínima (%)",
            "rain_max": "Precipitação Máxima (mm)",
            "rad_max": "Radiação Máxima (W/m²)",
            "wind_avg": "Velocidade Média do Vento (m/s)",
            "wind_max": "Velocidade Máxima do Vento (m/s)",
            "Area_Plantada": "Área Plantada (ha)",
            "Ano": "Ano",
        }
        corr_matrix = corr_matrix.rename(index=rename_dict, columns=rename_dict)

        # Plotar o mapa de calor
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,  # Exibe os valores nas células
            fmt=".2f",
            cmap="coolwarm",  # Paleta de cores
            cbar=True,
            square=True,  # Células quadradas
            linewidths=0.5,
        )
        plt.title(
            "Mapa de Calor da Correlação entre Variáveis Climáticas e Área Plantada",
            fontsize=14,
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Exibir o gráfico no Streamlit
        st.pyplot(plt)
        plt.close()
    except Exception as e:
        st.error(f"Erro ao gerar mapa de calor: {e}")


def plot_climatic_influence(correlations: pd.Series):
    """
    Plota as variáveis climáticas mais influentes com nomes mais descritivos.
    """
    # Renomear variáveis para facilitar a leitura
    rename_dict = {
        "temp_max": "Temperatura Máxima (°C)",
        "temp_avg": "Temperatura Média (°C)",
        "temp_min": "Temperatura Mínima (°C)",
        "hum_max": "Umidade Máxima (%)",
        "rain_max": "Precipitação Máxima (mm)",
        "rad_max": "Radiação Máxima (W/m²)",
        "wind_avg": "Velocidade Média do Vento (m/s)",
        "wind_max": "Velocidade Máxima do Vento (m/s)",
        "hum_min": "Umidade Mínima (%)",
        "Ano": "Ano",
        "Area_Plantada": "Área Plantada",
    }

    correlations = correlations.drop(
        "Area_Plantada", errors="ignore"
    )  # Remover redundância
    correlations = correlations.rename(index=rename_dict)  # Renomear variáveis
    correlations = correlations.sort_values(ascending=False)  # Ordenar por correlação

    # Criar o gráfico
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=correlations.values, y=correlations.index, hue=correlations.index, dodge=False
    )
    plt.title("Correlação entre Variáveis Climáticas e Área Plantada de Algodão")
    plt.xlabel("Correlação")
    plt.ylabel("Variáveis Climáticas")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Exibir o gráfico no Streamlit
    st.pyplot(plt)


def plot_historical_trends(historical_trends: pd.DataFrame):
    """
    Plota as tendências históricas na área plantada.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=historical_trends, x="Ano", y="Area_Planted")
    plt.title("Tendências Históricas da Área Plantada")
    plt.xlabel("Ano")
    plt.ylabel("Área Plantada (ha)")
    st.pyplot(plt)


def plot_scatter(cotton_data: pd.DataFrame, weather_data: pd.DataFrame):
    """
    Plota scatterplot das variáveis: temperatura média vs área plantada.
    """
    # Renomear colunas, se necessário
    cotton_data.rename(columns={"Area_Plantada": "Area_Planted"}, inplace=True)

    # Verificar colunas nos datasets
    required_cols = {"Ano", "Area_Planted"}
    if not required_cols.issubset(cotton_data.columns):
        raise ValueError(
            f"Faltando colunas no dataset de algodão: {required_cols - set(cotton_data.columns)}"
        )

    weather_cols = {"Ano", "temp_avg", "rain_max"}
    if not weather_cols.issubset(weather_data.columns):
        raise ValueError(
            f"Faltando colunas no dataset meteorológico: {weather_cols - set(weather_data.columns)}"
        )

    # Filtrar dados por anos em comum
    common_years = set(cotton_data["Ano"]).intersection(weather_data["Ano"])
    cotton_filtered = cotton_data[cotton_data["Ano"].isin(common_years)]
    weather_filtered = weather_data[weather_data["Ano"].isin(common_years)]

    # Fazer o merge dos dados
    combined_data = pd.merge(cotton_filtered, weather_filtered, on="Ano")

    # Amostrar dados para melhorar desempenho (exemplo: 20%)
    if len(combined_data) > 10000:
        combined_data = combined_data.sample(frac=0.2, random_state=42)

    # Gerar scatterplot
    plt.figure(figsize=(8, 5))
    plt.scatter(combined_data["temp_avg"], combined_data["Area_Planted"], alpha=0.7)
    plt.title("Dispersão: Temperatura Média vs Área Plantada")
    plt.xlabel("Temperatura Média (°C)")
    plt.ylabel("Área Plantada (ha)")
    plt.grid(True)
    st.pyplot(plt)


def plot_interactive_scatter(data, x_col, y_col, title, x_label, y_label):
    """
    Cria um gráfico de dispersão interativo com Plotly.
    """
    fig = px.scatter(data, x=x_col, y=y_col, title=title, labels={x_col: x_label, y_col: y_label})
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    st.plotly_chart(fig)

def plot_interactive_line(data, x_col, y_col, title, x_label, y_label):
    """
    Cria um gráfico de linha interativo com Plotly.
    """
    fig = px.line(data, x=x_col, y=y_col, title=title, labels={x_col: x_label, y_col: y_label})
    st.plotly_chart(fig)


def plot_historical_trends_with_prediction(historical_trends, predicted_areas):
    plt.figure(figsize=(10, 6))
    plt.plot(
        historical_trends["Ano"],
        historical_trends["Area_Planted"],
        label="Histórico",
        marker="o",
        color="blue",
    )
    plt.plot(
        predicted_areas["Ano"],
        predicted_areas["Area_Planted_Predicted"],
        label="Previsão",
        linestyle="--",
        color="orange",
    )
    plt.title("Tendências Históricas e Previsão da Área Plantada")
    plt.xlabel("Ano")
    plt.ylabel("Área Plantada (ha)")
    plt.legend()
    plt.grid()
    st.pyplot(plt)
    plt.close()
