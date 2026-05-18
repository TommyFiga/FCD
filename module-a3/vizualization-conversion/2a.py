import pandas as pd
import os

def exercicio_2a(caminho_ou_url, nome_dataset, tipo_problema):
    print(f"\n{'='*40}")
    print(f"Dataset: {nome_dataset}")
    print(f"{'='*40}")
    
    # Carregar o dataset
    df = pd.read_csv(caminho_ou_url)
    
    # (i) o número total de características
    total_caracteristicas = df.shape[1]
    
    # (ii) o número de características numéricas
    num_numericas = df.select_dtypes(include=['number']).shape[1]
    
    # (iii) o número de características categóricas
    num_categoricas = df.select_dtypes(exclude=['number']).shape[1]
    
    # (iv) a percentagem de valores omissos
    total_valores = df.shape[0] * df.shape[1]
    valores_omissos = df.isnull().sum().sum()
    percent_omissos = (valores_omissos / total_valores) * 100
    
    print(f"(i) Número total de características: {total_caracteristicas}")
    print(f"(ii) Número de características numéricas: {num_numericas}")
    print(f"(iii) Número de características categóricas: {num_categoricas}")
    print(f"(iv) Percentagem de valores omissos: {percent_omissos:.2f}%")
    
    # (v) o problema associado
    print(f"(v) Problema associado: {tipo_problema}")
    
    # Transformar para outro formato de ficheiro (de CSV para JSON)
    nome_ficheiro_saida = f"{nome_dataset.lower().replace(' ', '_')}_convertido.json"
    df.to_json(nome_ficheiro_saida, orient="records", indent=4)
    

if __name__ == "__main__":
    
    url_titanic = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    exercicio_2a(url_titanic, "Titanic", "Classificação")
    
    url_iris = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    exercicio_2a(url_iris, "Iris", "Classificação")
    
    url_tips = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    exercicio_2a(url_tips, "Tips", "Regressão")