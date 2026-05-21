"""
FCD - Módulo B.1 | Análise Exploratória de Dados (EDA)
Datasets:
  · Ames Housing        (Ames, Iowa, EUA | 2006–2010)
  · California Housing  (Estado da Califórnia, EUA | Censo 1990)
  · King County         (Condado de King, Washington, EUA | Mai 2014 – Mai 2015)
  · Melbourne Housing   (Melbourne, Victoria, Austrália | Jan 2016 – Set 2018)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ─── Configuração global de estilo ───────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ─── Utilitários ─────────────────────────────────────────────────────────────

def save_fig(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [gráfico] {name}")


def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


# ─── Carregamento dos dados ───────────────────────────────────────────────────

def load_ames() -> pd.DataFrame:
    """Ames, Iowa, EUA | 2006–2010"""
    path = os.path.join(DATA_DIR, "AmesHousing.csv")
    df = pd.read_csv(path)
    for c in ["SalePrice", "Sale Price", "sale_price"]:
        if c in df.columns:
            df = df.rename(columns={c: "SalePrice"})
            break
    return df


def load_california() -> pd.DataFrame:
    """Estado da Califórnia, EUA | Censo 1990"""
    path = os.path.join(DATA_DIR, "housing.csv")
    df = pd.read_csv(path)
    if "median_house_value" in df.columns:
        df = df.rename(columns={"median_house_value": "SalePrice"})
    return df


def load_king_county() -> pd.DataFrame | None:
    """Condado de King, Washington, EUA | Mai 2014 – Mai 2015"""
    path = os.path.join(DATA_DIR, "kc_house_data.csv")
    if not os.path.exists(path):
        print("  [AVISO] kc_house_data.csv não encontrado — King County ignorado.")
        return None
    df = pd.read_csv(path)
    if "price" in df.columns:
        df = df.rename(columns={"price": "SalePrice"})
    # Remove colunas sem valor preditivo
    for col in ["id", "date"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def load_melbourne() -> pd.DataFrame | None:
    """Melbourne, Victoria, Austrália | Jan 2016 – Set 2018"""
    
    path = os.path.join(DATA_DIR, "Melbourne_housing.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "Price" in df.columns:
            df = df.rename(columns={"Price": "SalePrice"})
        # Remove colunas de texto livre / alta cardinalidade
        drop_cols = ["Address", "Date", "SellerG", "Suburb",
                        "CouncilArea", "Regionname"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        # Remove linhas sem preço
        df = df.dropna(subset=["SalePrice"])
        return df
    print("  [AVISO] Melbourne_housing.csv não encontrado — Melbourne ignorado.")
    return None


# ─── EDA genérica (aplicada a cada dataset) ──────────────────────────────────

def run_eda(df: pd.DataFrame, name: str, target: str = "SalePrice"):
    """Executa EDA completa e exporta gráficos com prefixo `name`."""

    section(f"EDA — {name}")
    prefix = name.lower().replace(" ", "_")

    # 1. Visão geral
    print(f"\n  Shape: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"  Tipos de dados:\n{df.dtypes.value_counts().to_string()}")
    print(f"\n  Primeiras linhas:\n{df.head(3).to_string()}")

    # 2. Estatísticas descritivas (numéricas)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"\n  Colunas numéricas ({len(num_cols)}): {num_cols[:8]}{'...' if len(num_cols)>8 else ''}")
    print(f"  Colunas categóricas ({len(cat_cols)}): {cat_cols[:8]}{'...' if len(cat_cols)>8 else ''}")

    desc = df[num_cols].describe().T
    desc["cv"] = desc["std"] / desc["mean"]
    desc["missing%"] = df[num_cols].isnull().mean() * 100
    print(f"\n  Estatísticas numéricas (top 10 por missing%):\n"
          f"{desc.sort_values('missing%', ascending=False).head(10).round(2).to_string()}")

    # 3–10. Gráficos
    _plot_missing(df, name, prefix)
    _plot_target_distribution(df, target, name, prefix)
    _plot_histograms(df, num_cols, target, name, prefix)
    _plot_boxplots(df, num_cols, target, name, prefix)
    _plot_correlation_target(df, num_cols, target, name, prefix)
    _plot_heatmap(df, num_cols, target, name, prefix)
    _plot_scatter_top(df, num_cols, target, name, prefix)
    if cat_cols:
        _plot_categorical(df, cat_cols, target, name, prefix)

    _print_missing_summary(df, name)


# ─── Funções de gráficos ─────────────────────────────────────────────────────

def _plot_missing(df: pd.DataFrame, name: str, prefix: str):
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("  Sem valores em falta.")
        return

    top = missing.head(30)
    fig, ax = plt.subplots(figsize=(12, max(4, len(top) * 0.35)))
    colors = ["#e74c3c" if v > 40 else "#e67e22" if v > 20 else "#3498db" for v in top.values]
    bars = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], height=0.7)
    ax.set_xlabel("% de valores em falta")
    ax.set_title(f"{name} — Valores em falta por coluna", fontweight="bold")
    ax.axvline(5, ls="--", lw=0.8, color="gray", alpha=0.6, label="5%")
    ax.axvline(20, ls="--", lw=0.8, color="orange", alpha=0.6, label="20%")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    save_fig(f"{prefix}_missing_values.png")


def _plot_target_distribution(df: pd.DataFrame, target: str, name: str, prefix: str):
    if target not in df.columns:
        return
    vals = df[target].dropna()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(vals, bins=50, color="#3498db", edgecolor="white", linewidth=0.3)
    axes[0].set_title("Distribuição original")
    axes[0].set_xlabel(target)
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    log_vals = np.log1p(vals)
    axes[1].hist(log_vals, bins=50, color="#2ecc71", edgecolor="white", linewidth=0.3)
    axes[1].set_title("Distribuição log(1+x)")
    axes[1].set_xlabel(f"log({target})")

    axes[2].boxplot(vals, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#3498db", alpha=0.6))
    axes[2].set_title("Boxplot")
    axes[2].set_ylabel(target)
    axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle(f"{name} — Variável alvo: {target}", fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(f"{prefix}_target_distribution.png")

    print(f"\n  [{target}] min={vals.min():,.0f} | median={vals.median():,.0f} "
          f"| mean={vals.mean():,.0f} | max={vals.max():,.0f} | skew={vals.skew():.2f}")


def _plot_histograms(df: pd.DataFrame, num_cols: list, target: str, name: str, prefix: str):
    cols = [c for c in num_cols if c != target][:12]
    if not cols:
        return
    ncols = 4
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=40, color="#3498db",
                     edgecolor="white", linewidth=0.3, alpha=0.85)
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{name} — Distribuição das variáveis numéricas", fontweight="bold")
    plt.tight_layout()
    save_fig(f"{prefix}_histograms.png")


def _plot_boxplots(df: pd.DataFrame, num_cols: list, target: str, name: str, prefix: str):
    cols = [c for c in num_cols if c != target][:12]
    if not cols:
        return
    ncols = 4
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#e74c3c", alpha=0.5),
                        flierprops=dict(marker="o", markersize=2, alpha=0.3))
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{name} — Boxplots (detecção de outliers)", fontweight="bold")
    plt.tight_layout()
    save_fig(f"{prefix}_boxplots.png")


def _plot_correlation_target(df: pd.DataFrame, num_cols: list, target: str, name: str, prefix: str):
    if target not in df.columns:
        return
    corr = df[num_cols].corr()[target].drop(target).sort_values()
    selected = pd.concat([corr.head(10), corr.tail(10)])

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in selected.values]
    bars = ax.barh(selected.index, selected.values, color=colors, height=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Correlação de Pearson")
    ax.set_title(f"{name} — Correlação com {target} (top 20)", fontweight="bold")
    for bar, val in zip(bars, selected.values):
        x = val + 0.01 if val >= 0 else val - 0.01
        ha = "left" if val >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha=ha, fontsize=8)
    plt.tight_layout()
    save_fig(f"{prefix}_correlation_target.png")
    print(f"\n  Top 5 correlação positiva com {target}:")
    print(corr.tail(5).round(3).to_string())
    print(f"  Top 5 correlação negativa com {target}:")
    print(corr.head(5).round(3).to_string())


def _plot_heatmap(df: pd.DataFrame, num_cols: list, target: str, name: str, prefix: str):
    if target not in df.columns:
        return
    corr_full = df[num_cols].corr()[target].drop(target).abs()
    top_cols  = corr_full.nlargest(15).index.tolist() + [target]
    corr_matrix = df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", ax=ax,
                cmap="RdYlGn", center=0, linewidths=0.4, linecolor="white",
                annot_kws={"size": 7}, vmin=-1, vmax=1)
    ax.set_title(f"{name} — Matriz de correlação (top 15 features + target)", fontweight="bold")
    plt.tight_layout()
    save_fig(f"{prefix}_heatmap_correlation.png")


def _plot_scatter_top(df: pd.DataFrame, num_cols: list, target: str, name: str, prefix: str):
    if target not in df.columns:
        return
    corr = df[num_cols].corr()[target].drop(target).abs()
    top4 = corr.nlargest(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    for i, col in enumerate(top4):
        data = df[[col, target]].dropna()
        axes[i].scatter(data[col], data[target], alpha=0.3, s=8,
                        color="#3498db", edgecolors="none")
        z = np.polyfit(data[col], data[target], 1)
        x_line = np.linspace(data[col].min(), data[col].max(), 200)
        axes[i].plot(x_line, np.poly1d(z)(x_line), "r-", lw=1.5, alpha=0.8)
        r = data[[col, target]].corr().iloc[0, 1]
        axes[i].set_title(f"{col}  (r = {r:.3f})", fontsize=10)
        axes[i].set_xlabel(col, fontsize=9)
        axes[i].set_ylabel(target, fontsize=9)
        axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.suptitle(f"{name} — Scatter: top 4 preditores vs {target}", fontweight="bold")
    plt.tight_layout()
    save_fig(f"{prefix}_scatter_top_features.png")


def _plot_categorical(df: pd.DataFrame, cat_cols: list, target: str, name: str, prefix: str):
    useful = [c for c in cat_cols if 2 <= df[c].nunique() <= 20 and target in df.columns][:6]
    if not useful:
        return
    ncols = 2
    nrows = (len(useful) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()
    for i, col in enumerate(useful):
        order = df.groupby(col)[target].median().sort_values(ascending=False).index
        sns.boxplot(data=df[[col, target]].dropna(), x=col, y=target,
                    order=order, ax=axes[i], palette="Blues_r", linewidth=0.8)
        axes[i].set_title(f"{target} por {col}", fontsize=10)
        axes[i].tick_params(axis="x", rotation=30, labelsize=8)
        axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{name} — Variáveis categóricas vs {target}", fontweight="bold")
    plt.tight_layout()
    save_fig(f"{prefix}_categorical_vs_target.png")


def _print_missing_summary(df: pd.DataFrame, name: str):
    total   = df.isnull().sum()
    pct     = df.isnull().mean() * 100
    summary = pd.DataFrame({"missing_count": total, "missing_pct": pct})
    summary = summary[summary["missing_count"] > 0].sort_values("missing_pct", ascending=False)
    if summary.empty:
        print(f"\n  [{name}] Sem valores em falta — dataset completo.")
    else:
        print(f"\n  [{name}] Resumo de missing values ({len(summary)} colunas afectadas):")
        print(summary.head(15).round(2).to_string())


# ─── Análise geográfica ───────────────────────────────────────────────────────

def plot_geo(df: pd.DataFrame, title: str, prefix: str):
    """Mapa de preços para datasets com latitude/longitude."""
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return
    if "SalePrice" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 9))
    sc = ax.scatter(df["longitude"], df["latitude"],
                    c=df["SalePrice"], cmap="RdYlGn_r",
                    alpha=0.4, s=5, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("SalePrice", fontsize=10)
    cbar.formatter = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    cbar.update_ticks()
    ax.set_title(f"{title} — Distribuição geográfica dos preços", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    save_fig(f"{prefix}_geo_price_map.png")


# ─── Comparação entre os 4 datasets ──────────────────────────────────────────

def compare_datasets(datasets: dict):
    section("Comparação resumida entre os 4 datasets")
    rows = []
    for name, df in datasets.items():
        if df is None or "SalePrice" not in df.columns:
            continue
        vals = df["SalePrice"].dropna()
        rows.append({
            "Dataset":       name,
            "Linhas":        df.shape[0],
            "Colunas":       df.shape[1],
            "Missing %":     f"{df.isnull().mean().mean() * 100:.1f}%",
            "Price min":     f"{vals.min():,.0f}",
            "Price median":  f"{vals.median():,.0f}",
            "Price mean":    f"{vals.mean():,.0f}",
            "Price max":     f"{vals.max():,.0f}",
            "Skewness":      f"{vals.skew():.2f}",
        })
    comp = pd.DataFrame(rows).set_index("Dataset")
    print(f"\n{comp.to_string()}")

    # Distribuição comparativa normalizada
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
    for (name, df), color in zip(datasets.items(), colors):
        if df is None or "SalePrice" not in df.columns:
            continue
        vals = df["SalePrice"].dropna()
        vals_norm = (vals - vals.min()) / (vals.max() - vals.min())
        ax.hist(vals_norm, bins=60, alpha=0.5, label=name, color=color,
                density=True, edgecolor="white", lw=0.2)
    ax.set_xlabel("Preço normalizado [0, 1]")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição comparativa de preços (normalizada)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_fig("comparison_price_distribution.png")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("FCD B.1 — EDA: 4 datasets imobiliários")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}\n")

    # ── Carregamento ──────────────────────────────────────────────────────────
    print("A carregar datasets...")
    ames        = load_ames()
    california  = load_california()
    king_county = load_king_county()
    melbourne   = load_melbourne()

    # Dicionário ordenado — mantém a ordem para a comparação final
    datasets = {
        "Ames Housing":       ames,
        "California Housing": california,
        "King County":        king_county,
        "Melbourne Housing":  melbourne,
    }

    for name, df in datasets.items():
        if df is not None:
            print(f"  {name:<22}: {df.shape[0]:>6,} linhas × {df.shape[1]} colunas")

    # ── EDA individual ────────────────────────────────────────────────────────
    for name, df in datasets.items():
        if df is not None:
            run_eda(df, name, target="SalePrice")

    # ── Mapas geográficos (datasets com lat/lon) ──────────────────────────────
    geo_datasets = {
        "California Housing": (california, "california_housing"),
        "King County":        (king_county, "king_county"),
    }
    for name, (df, prefix) in geo_datasets.items():
        if df is not None:
            plot_geo(df, name, prefix)

    # ── Comparação global ─────────────────────────────────────────────────────
    compare_datasets(datasets)

    section("EDA concluída")
    print(f"  Todos os gráficos guardados em: {os.path.abspath(OUTPUT_DIR)}")
    print("  Ficheiros gerados:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    · {f}")

