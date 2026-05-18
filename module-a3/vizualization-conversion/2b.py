import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def exercicio_2b():
    print("A carregar o dataset Iris...")
    url_iris = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url_iris)
    
    # (i) Scatter-plot entre três pares de variáveis
    
    sns.pairplot(df, vars=['sepal_length', 'sepal_width', 'petal_length'], hue='species')
    plt.suptitle("Scatter-plot - Relação entre medidas das Flores Iris", y=1.02)
    plt.show() # O programa pausa aqui até fechares a janela do gráfico
    
    # (ii) Gráfico de barras com o número de instâncias por classe
    print("A gerar o Gráfico de Barras...")
    
    sns.countplot(data=df, x='species', palette='Set2')
    plt.title("Número de Instâncias por Classe (Espécie)")
    plt.xlabel("Espécie da Flor")
    plt.ylabel("Quantidade")
    plt.show()

if __name__ == "__main__":
    exercicio_2b()