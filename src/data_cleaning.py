import pandas as pd


def load_cotton_data(filepath: str) -> pd.DataFrame:
    """
    Carrega e processa os dados de algodão do arquivo Excel.
    """
    try:
        data = pd.read_excel(filepath, engine="openpyxl", skiprows=4, header=None)
        col_count = len(data.columns)

        # Criar nomes de colunas dinamicamente
        col_names = ["Região/UF"] + [
            str(year) for year in range(1976, 1976 + col_count - 1)
        ]
        data.columns = col_names[:col_count]

        # Excluir totais e valores agregados
        data = data[~data["Região/UF"].str.contains("BRASIL|NORTE/NORDESTE", na=False)]

        # Transformar para formato longo
        data_long = data.melt(
            id_vars=["Região/UF"], var_name="Ano", value_name="Area_Plantada"
        )
        data_long["Area_Plantada"] = pd.to_numeric(
            data_long["Area_Plantada"], errors="coerce"
        )
        data_long["Ano"] = pd.to_numeric(
            data_long["Ano"].str.extract(r"(\d{4})")[0], errors="coerce"
        )

        # Remover valores ausentes
        data_long.dropna(subset=["Ano", "Area_Plantada"], inplace=True)

        return data_long
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar dados de algodão: {e}")


def load_weather_data(filepath: str) -> pd.DataFrame:
    """
    Carrega e processa os dados climáticos.
    """
    try:
        # Carregar os dados
        data = pd.read_csv(filepath)

        # Converter a coluna DATA para datetime
        data["DATA"] = pd.to_datetime(data["DATA (YYYY-MM-DD)"], errors="coerce")
        data["Ano"] = data["DATA"].dt.year
        data["Mes"] = data["DATA"].dt.month

        # Definir estações do ano com base nos meses
        season_bins = [0, 2, 5, 8, 11, 12]  # Divisão em meses
        season_labels = ["Verão", "Outono", "Inverno", "Primavera", "Verão"]
        data["Estacao"] = pd.cut(
            data["Mes"],
            bins=season_bins,
            labels=season_labels,
            ordered=False,
            right=True,
        )

        # Verificar duplicatas ou problemas
        if data["Estacao"].isna().any():
            raise ValueError(
                "Erro ao mapear meses para estações: valores nulos detectados."
            )

        print("Pré-visualização dos dados meteorológicos:")
        print(data.head())

        return data
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar dados meteorológicos: {e}")
