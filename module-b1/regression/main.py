"""
FCD - Módulo B.1 | Parte 2: Pré-processamento, Modelação e Avaliação
Datasets : Ames Housing, California Housing, King County, Melbourne Housing
Técnicas  : MLP (sklearn), Rede RBF (implementação própria), Random Forest
Métricas  : MAE, RMSE, R², MAPE
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ─── Configuração ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.2        # 80/20 split
N_CENTERS    = 50         # centros da rede RBF (ajustável)

MODEL_LABELS = ["MLP", "RBF Network", "Random Forest"]
COLORS       = ["#3498db", "#e67e22", "#2ecc71"]


# ─── Utilitários ─────────────────────────────────────────────────────────────

def save_fig(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [gráfico] {name}")


def section(title: str):
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (evita divisão por zero)."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    if label:
        print(f"    {label:<18}  MAE={mae:>12,.0f}  RMSE={rmse:>12,.0f}  "
              f"R²={r2:>6.4f}  MAPE={mape_val:>6.2f}%")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape_val}


# ─── Carregamento dos datasets ───────────────────────────────────────────────

DATASET_CONFIGS = {
    "Ames Housing": {
        "file": "AmesHousing.csv",
        "target_candidates": ["SalePrice", "Sale Price", "sale_price"],
        "drop_cols": ["Order", "PID"],          # colunas ID sem valor preditivo
        "geo": "Ames, Iowa, EUA",
        "period": "2006–2010",
    },
    "California Housing": {
        "file": "housing.csv",
        "target_candidates": ["median_house_value", "SalePrice"],
        "drop_cols": ["ocean_proximity"],       # categórica simples — mantida via encoding
        "geo": "Estado da Califórnia, EUA",
        "period": "Censo 1990",
    },
    "King County": {
        "file": "kc_house_data.csv",            # nome típico do Kaggle
        "target_candidates": ["price"],
        "drop_cols": ["id", "date"],
        "geo": "Condado de King, Washington, EUA",
        "period": "Mai 2014 – Mai 2015",
    },
    "Melbourne Housing": {
        "file": "Melbourne_housing.csv",   # nome típico do Kaggle
        "target_candidates": ["Price"],
        "drop_cols": ["Address", "Date", "SellerG", "Suburb", "CouncilArea", "Regionname"],
        "geo": "Melbourne, Victoria, Austrália",
        "period": "Jan 2016 – Set 2018",
    },
}


def load_dataset(name: str) -> pd.DataFrame | None:
    cfg  = DATASET_CONFIGS[name]
    path = os.path.join(DATA_DIR, cfg["file"])
    if not os.path.exists(path):
        print(f"  [AVISO] Ficheiro não encontrado: {cfg['file']} — dataset ignorado.")
        return None

    df = pd.read_csv(path)

    # Normaliza coluna alvo para "SalePrice"
    for cand in cfg["target_candidates"]:
        if cand in df.columns and cand != "SalePrice":
            df = df.rename(columns={cand: "SalePrice"})
            break

    if "SalePrice" not in df.columns:
        print(f"  [AVISO] Coluna alvo não encontrada em {name} — dataset ignorado.")
        return None

    # Remove linhas sem preço
    df = df.dropna(subset=["SalePrice"])
    df = df[df["SalePrice"] > 0]

    # Remove colunas ID/datas desnecessárias
    drop = [c for c in cfg.get("drop_cols", []) if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    print(f"  {name:<22}: {df.shape[0]:>6,} linhas × {df.shape[1]} colunas "
          f"| {cfg['geo']} | {cfg['period']}")
    return df


# ─── Pré-processamento ────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, target: str = "SalePrice"):
    """
    Pipeline de pré-processamento:
      1. Separa features / target
      2. Encoding ordinal de categóricas (label encoding)
      3. Imputação de missing values (mediana para numéricas, moda para categóricas)
      4. Standardização (média 0, desvio 1)
      5. Split treino/teste estratificado por quintil de preço
    Retorna: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    y = df[target].values.astype(float)
    X_raw = df.drop(columns=[target])

    # Encoding de categóricas
    cat_cols = X_raw.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        X_raw[col] = pd.Categorical(X_raw[col]).codes.astype(float)
        X_raw[col] = X_raw[col].replace(-1, np.nan)   # -1 era NaN

    feature_names = X_raw.columns.tolist()
    X = X_raw.values.astype(float)

    # Imputação (mediana)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Standardização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_names


# ─── Rede RBF (implementação própria) ────────────────────────────────────────

class RBFNetwork:
    """
    Rede neuronal de base radial (Radial Basis Function Network).

    Arquitectura:
      • Camada de entrada  : n_features neurónios
      • Camada oculta RBF  : n_centers neurónios com função de activação Gaussiana
        φ_j(x) = exp(-||x - μ_j||² / (2σ_j²))
      • Camada de saída    : 1 neurónio com activação linear (Ridge regression)

    Treino:
      1. Centros μ_j determinados por K-Means (aprendizagem não supervisionada)
      2. Larguras σ_j = d_max / sqrt(2·k)  (heurística P-nearest)
      3. Pesos de saída por Ridge regression (regularização L2)
    """

    def __init__(self, n_centers: int = 50, alpha: float = 1.0, random_state: int = 42):
        self.n_centers    = n_centers
        self.alpha        = alpha          # regularização Ridge
        self.random_state = random_state
        self.centers_     = None
        self.sigma_       = None
        self.linear_      = Ridge(alpha=alpha)

    def _rbf_features(self, X: np.ndarray) -> np.ndarray:
        """Calcula a matrix de activações φ ∈ R^(n_samples × n_centers)."""
        diff = X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]   # (n, k, d)
        dist2 = np.sum(diff ** 2, axis=-1)                              # (n, k)
        return np.exp(-dist2 / (2 * self.sigma_ ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 1. Centros via K-Means
        km = KMeans(n_clusters=self.n_centers, random_state=self.random_state,
                    n_init=5, max_iter=200)
        km.fit(X)
        self.centers_ = km.cluster_centers_                             # (k, d)

        # 2. Largura: heurística  σ = d_max / sqrt(2k)
        from scipy.spatial.distance import cdist
        dists = cdist(self.centers_, self.centers_)
        d_max = dists.max()
        self.sigma_ = d_max / np.sqrt(2 * self.n_centers)
        if self.sigma_ == 0:
            self.sigma_ = 1.0

        # 3. Pesos de saída (Ridge)
        Phi = self._rbf_features(X)
        self.linear_.fit(Phi, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Phi = self._rbf_features(X)
        return self.linear_.predict(Phi)

    
# ─── Configuração dos modelos ─────────────────────────────────────────────────

def build_models(n_features: int):
    """
    Devolve dict com os três modelos configurados.
    Topologia MLP: [n_features] → [128, 64, 32] → [1]
    Topologia RBF: [n_features] → [50 RBF] → [1]
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )

    rbf = RBFNetwork(n_centers=N_CENTERS, alpha=1.0, random_state=RANDOM_STATE)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return {"MLP": mlp, "RBF Network": rbf, "Random Forest": rf}


def topology_summary(models: dict, n_features: int) -> None:
    print("\n  ── Topologia dos modelos ──")
    print(f"  MLP           : Entrada({n_features}) → Dense(128,ReLU) → Dense(64,ReLU) "
          f"→ Dense(32,ReLU) → Saída(1,linear)")
    # Substituímos a chamada ao método da classe por uma string direta com os valores
    print(f"  RBF Network   : Entrada({n_features}) → Camada RBF: {N_CENTERS} neurónios Gaussianos → Saída: 1 (linear)")
    print(f"  Random Forest : 200 árvores, max_features=sqrt, min_samples_leaf=2")


# ─── Importância de variáveis ─────────────────────────────────────────────────

def feature_importances(model, X_test, y_test, feature_names, model_name, dataset_name, prefix):
    """Calcula e plota importância de variáveis (permutation importance)."""
    pi = permutation_importance(model, X_test, y_test,
                                n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=feature_names).sort_values(ascending=False)
    top = imp.head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLORS[MODEL_LABELS.index(model_name)]] * len(top)
    ax.barh(top.index[::-1], top.values[::-1], color=colors, height=0.7)
    ax.set_xlabel("Importância (redução de R² por permutação)")
    ax.set_title(f"{dataset_name} — {model_name}: Top 15 variáveis", fontweight="bold")
    plt.tight_layout()
    save_fig(f"{prefix}_{model_name.lower().replace(' ','_')}_feature_importance.png")

    print(f"\n  Top 5 variáveis ({model_name}):")
    for feat, val in imp.head(5).items():
        print(f"    {feat:<30}  {val:.4f}")
    return imp


# ─── Gráficos comparativos ────────────────────────────────────────────────────

def plot_metrics_comparison(results: dict, dataset_name: str, prefix: str):
    """Barchart comparativo das métricas pelos 3 modelos."""
    metrics = ["MAE", "RMSE", "R2", "MAPE"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        vals  = [results[m][metric] for m in MODEL_LABELS]
        bars  = ax.bar(MODEL_LABELS, vals, color=COLORS, width=0.5, edgecolor="white")
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel(metric)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.4f}" if metric in ("R2",) else (
                  f"{val:.2f}%" if metric == "MAPE" else f"{val:,.0f}")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    fmt, ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=10)

    fig.suptitle(f"{dataset_name} — Comparação de métricas", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(f"{prefix}_metrics_comparison.png")


def plot_predictions(y_test, preds: dict, dataset_name: str, prefix: str):
    """Scatter plot real vs previsto para os 3 modelos."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (model_name, y_pred), color in zip(axes, preds.items(), COLORS):
        lim_min = min(y_test.min(), y_pred.min())
        lim_max = max(y_test.max(), y_pred.max())
        ax.scatter(y_test, y_pred, alpha=0.25, s=6, color=color, edgecolors="none")
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1, alpha=0.7)
        r2 = r2_score(y_test, y_pred)
        ax.set_title(f"{model_name}  (R²={r2:.3f})", fontsize=10)
        ax.set_xlabel("Real")
        ax.set_ylabel("Previsto")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    fig.suptitle(f"{dataset_name} — Real vs Previsto", fontweight="bold")
    plt.tight_layout()
    save_fig(f"{prefix}_real_vs_predicted.png")


def plot_residuals(y_test, preds: dict, dataset_name: str, prefix: str):
    """Distribuição dos resíduos por modelo."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (model_name, y_pred), color in zip(axes, preds.items(), COLORS):
        residuals = y_test - y_pred
        ax.hist(residuals, bins=50, color=color, alpha=0.75, edgecolor="white", lw=0.3)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_title(f"{model_name}", fontsize=10)
        ax.set_xlabel("Resíduo (real − previsto)")
        ax.set_ylabel("Frequência")
    fig.suptitle(f"{dataset_name} — Distribuição dos resíduos", fontweight="bold")
    plt.tight_layout()
    save_fig(f"{prefix}_residuals.png")


# ─── Pipeline principal por dataset ──────────────────────────────────────────

def run_dataset(name: str, df: pd.DataFrame, all_results: list):
    section(f"MODELAÇÃO — {name}")
    prefix = name.lower().replace(" ", "_")

    # Pré-processamento
    print("\n  A pré-processar...")
    X_train, X_test, y_train, y_test, scaler, feat_names = preprocess(df)
    print(f"  Treino: {X_train.shape} | Teste: {X_test.shape}")
    print(f"  Features: {len(feat_names)}")

    # Modelos
    models  = build_models(X_train.shape[1])
    topology_summary(models, X_train.shape[1])

    results  = {}
    preds    = {}
    best_r2  = -np.inf
    best_model_name = ""

    print("\n  ── Treino e avaliação ──")
    for model_name, model in models.items():
        print(f"\n  [{model_name}] A treinar...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds[model_name] = y_pred
        metrics = compute_metrics(y_test, y_pred, label=model_name)
        results[model_name] = metrics
        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model_name = model_name

    print(f"\n  ★ Melhor modelo em {name}: {best_model_name}  (R²={best_r2:.4f})")

    # Importância de variáveis — melhor modelo
    print(f"\n  A calcular importância de variáveis ({best_model_name})...")
    best_model = models[best_model_name]
    feature_importances(best_model, X_test, y_test, feat_names,
                        best_model_name, name, prefix)

    # Gráficos
    plot_metrics_comparison(results, name, prefix)
    plot_predictions(y_test, preds, name, prefix)
    plot_residuals(y_test, preds, name, prefix)

    # Guarda para resumo global
    for model_name, metrics in results.items():
        all_results.append({
            "Dataset": name,
            "Modelo": model_name,
            **metrics,
        })

    return results, best_model_name


# ─── Comparação global entre datasets ────────────────────────────────────────

def plot_global_comparison(all_results: list):
    section("COMPARAÇÃO GLOBAL")
    df_res = pd.DataFrame(all_results)
    print(f"\n{df_res.round(3).to_string(index=False)}")

    # Heatmap R² por dataset × modelo
    pivot = df_res.pivot(index="Dataset", columns="Modelo", values="R2")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                annot_kws={"size": 10}, ax=ax)
    ax.set_title("R² por dataset e modelo", fontweight="bold")
    plt.tight_layout()
    save_fig("global_r2_heatmap.png")

    # RMSE normalizado (% do preço mediano por dataset — para comparar entre datasets)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df_res["Dataset"].unique()))
    width = 0.25
    datasets = df_res["Dataset"].unique()
    for i, (model_name, color) in enumerate(zip(MODEL_LABELS, COLORS)):
        vals = df_res[df_res["Modelo"] == model_name]["RMSE"].values
        ax.bar(x + i * width, vals, width, label=model_name, color=color, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets, rotation=10, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE por dataset e modelo", fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    save_fig("global_rmse_comparison.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("FCD B.1 — Parte 2: Modelação e Avaliação")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}\n")

    print("A carregar datasets...")
    datasets = {}
    for name in DATASET_CONFIGS:
        df = load_dataset(name)
        if df is not None:
            datasets[name] = df

    if not datasets:
        print("Nenhum dataset carregado. Verifica os ficheiros em /data.")
        exit(1)

    all_results = []
    summary     = {}

    for name, df in datasets.items():
        results, best = run_dataset(name, df, all_results)
        summary[name] = best

    # Comparação global
    if len(all_results) > 0:
        plot_global_comparison(all_results)

    # Sumário final
    section("SUMÁRIO")
    print("\n  Dataset               | Melhor modelo")
    print("  " + "-" * 42)
    for name, best in summary.items():
        print(f"  {name:<22}| {best}")

    print(f"\n  Todos os gráficos guardados em: {os.path.abspath(OUTPUT_DIR)}")