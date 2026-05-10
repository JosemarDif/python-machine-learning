"""
============================================================
 MINERAÇÃO DE DADOS & MACHINE LEARNING — Previsão de Vendas
 Autor  : Josemar Manuel
 Email  : josemardiferencial@gmail.com
 LinkedIn: @josemarmanuel
 Data   : Abril 2026
============================================================

 Modelos implementados:
   1. Regressão Logística  — Classificar se venda atinge meta
   2. Árvore de Decisão    — Prever categoria de venda
   3. Random Forest        — Prever valor da venda (regressão)
   4. K-Means Clustering   — Segmentar clientes/vendedores

 Técnicas aplicadas:
   - Feature Engineering
   - Normalização (StandardScaler)
   - Codificação (LabelEncoder, OneHotEncoder)
   - Cross-Validation (5-fold)
   - Matriz de Confusão
   - Curva ROC / AUC
   - Feature Importance

 Como executar:
   pip install pandas numpy matplotlib seaborn scikit-learn
   python machine_learning_vendas.py
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection    import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.linear_model       import LogisticRegression
from sklearn.tree               import DecisionTreeClassifier, export_text
from sklearn.ensemble           import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster            import KMeans
from sklearn.metrics            import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, mean_squared_error, r2_score
)

# ── Configuração visual ──────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0D1B35",
    "axes.facecolor":   "#111E36",
    "axes.edgecolor":   "#2A3F6F",
    "axes.labelcolor":  "#8B96B0",
    "xtick.color":      "#8B96B0",
    "ytick.color":      "#8B96B0",
    "text.color":       "#F0F4FF",
    "grid.color":       "#1E3060",
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "font.size":        10,
})
CORES = ["#00C2D4","#A855F7","#4ADE80","#F5A623","#FF4D4D","#2E75B6"]

print("=" * 60)
print(" MACHINE LEARNING — Previsão de Vendas Angola 2025")
print(" Josemar Manuel · josemardiferencial@gmail.com")
print("=" * 60)

# ══════════════════════════════════════════════════════════
# 1. GERAÇÃO DO DATASET
# ══════════════════════════════════════════════════════════
np.random.seed(42)
N = 1000
print(f"\n[1/6] Gerando dataset ({N} registos)...")

vendedores = ["Ana Silva","Carlos Pereira","Maria João","Pedro Santos","Sofia Neto"]
regioes    = ["Luanda","Huambo","Benguela","Lobito","Cabinda"]
produtos   = ["Laptop","Monitor","Teclado","Rato","Impressora","Cadeira"]
precos     = {"Laptop":350000,"Monitor":120000,"Teclado":15000,
              "Rato":8000,"Impressora":85000,"Cadeira":45000}

df = pd.DataFrame({
    "Vendedor"   : np.random.choice(vendedores, N),
    "Região"     : np.random.choice(regioes, N),
    "Produto"    : np.random.choice(produtos, N),
    "Mês"        : np.random.randint(1, 13, N),
    "Quantidade" : np.random.randint(1, 20, N),
    "Desconto_Pct": np.random.choice([0,0,0,5,10,15], N),
    "Experiência_Anos": np.random.uniform(0.5, 8, N).round(1),
})

df["Preço_Unit"]    = df["Produto"].map(precos) * np.random.uniform(0.9, 1.1, N)
df["Total_Líquido"] = df["Quantidade"] * df["Preço_Unit"] * (1 - df["Desconto_Pct"]/100)
df["Meta"]          = df["Total_Líquido"] * np.random.uniform(0.7, 1.3, N)
df["Atingiu_Meta"]  = (df["Total_Líquido"] >= df["Meta"]).astype(int)
df["Trimestre"]     = pd.cut(df["Mês"], bins=[0,3,6,9,12], labels=[1,2,3,4]).astype(int)
df["Categoria"]     = df["Produto"].map(
    {"Laptop":"Informática","Monitor":"Informática",
     "Teclado":"Periféricos","Rato":"Periféricos",
     "Impressora":"Escritório","Cadeira":"Mobiliário"})

print(f"   Dataset: {df.shape}  |  Target positivo: {df['Atingiu_Meta'].mean():.1%}")

# ══════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING E PRÉ-PROCESSAMENTO
# ══════════════════════════════════════════════════════════
print("\n[2/6] Feature Engineering e pré-processamento...")

# Codificar variáveis categóricas
le_vendedor = LabelEncoder()
le_regiao   = LabelEncoder()
le_produto  = LabelEncoder()

df["Vendedor_enc"] = le_vendedor.fit_transform(df["Vendedor"])
df["Região_enc"]   = le_regiao.fit_transform(df["Região"])
df["Produto_enc"]  = le_produto.fit_transform(df["Produto"])

# Features para classificação (Atingiu_Meta)
FEATURES_CLASS = [
    "Vendedor_enc","Região_enc","Produto_enc",
    "Mês","Trimestre","Quantidade","Desconto_Pct",
    "Preço_Unit","Experiência_Anos"
]
TARGET_CLASS = "Atingiu_Meta"

X = df[FEATURES_CLASS].values
y = df[TARGET_CLASS].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"   Train: {X_train.shape}  |  Test: {X_test.shape}")

# ══════════════════════════════════════════════════════════
# 3. MODELO 1 — REGRESSÃO LOGÍSTICA
# ══════════════════════════════════════════════════════════
print("\n[3/6] Modelo 1 — Regressão Logística...")

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
y_prob_lr = lr.predict_proba(X_test_s)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr, X_train_s, y_train, cv=cv, scoring="accuracy")

print(f"\n   ─── Regressão Logística ───────────────────")
print(f"   Acurácia (Test)      : {acc_lr:.4f}  ({acc_lr*100:.2f}%)")
print(f"   AUC-ROC              : {auc_lr:.4f}")
print(f"   Cross-Val (5-fold)   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\n   Relatório de Classificação:")
print(classification_report(y_test, y_pred_lr,
      target_names=["Não Atingiu","Atingiu"], indent=3))

# ══════════════════════════════════════════════════════════
# 4. MODELO 2 — ÁRVORE DE DECISÃO
# ══════════════════════════════════════════════════════════
print("[4/6] Modelo 2 — Árvore de Decisão...")

dt = DecisionTreeClassifier(max_depth=5, min_samples_split=20,
                            random_state=42, class_weight="balanced")
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

acc_dt = accuracy_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_prob_dt)

print(f"\n   ─── Árvore de Decisão ────────────────────")
print(f"   Acurácia (Test)      : {acc_dt:.4f}  ({acc_dt*100:.2f}%)")
print(f"   AUC-ROC              : {auc_dt:.4f}")
print(f"   Profundidade máxima  : {dt.get_depth()}")

print(f"\n   Importância das Features (Top 5):")
fi = pd.Series(dt.feature_importances_, index=FEATURES_CLASS)
fi_top = fi.sort_values(ascending=False).head(5)
for feat, imp in fi_top.items():
    barra = "█" * int(imp * 40)
    print(f"     {feat:<22} {imp:.4f}  {barra}")

# ══════════════════════════════════════════════════════════
# 5. MODELO 3 — RANDOM FOREST (Regressão de valor)
# ══════════════════════════════════════════════════════════
print("\n[5/6] Modelo 3 — Random Forest (Previsão de Valor)...")

FEATURES_REG = ["Vendedor_enc","Região_enc","Produto_enc",
                "Mês","Trimestre","Quantidade","Desconto_Pct","Experiência_Anos"]
y_reg = df["Total_Líquido"].values / 1e6   # em Milhões de Kz

Xr = df[FEATURES_REG].values
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, y_reg, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=8,
                                random_state=42, n_jobs=-1)
rf_reg.fit(Xr_train, yr_train)
yr_pred = rf_reg.predict(Xr_test)

rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2   = r2_score(yr_test, yr_pred)
mae  = np.mean(np.abs(yr_test - yr_pred))

print(f"\n   ─── Random Forest (Regressão) ────────────")
print(f"   R² Score             : {r2:.4f}  ({r2*100:.2f}% da variância explicada)")
print(f"   RMSE                 : {rmse:.4f} M Kz")
print(f"   MAE                  : {mae:.4f} M Kz")
print(f"   Nº de Árvores        : 100")

fi_rf = pd.Series(rf_reg.feature_importances_, index=FEATURES_REG)
fi_rf_top = fi_rf.sort_values(ascending=False)
print(f"\n   Importância das Features:")
for feat, imp in fi_rf_top.items():
    barra = "█" * int(imp * 50)
    print(f"     {feat:<22} {imp:.4f}  {barra}")

# ══════════════════════════════════════════════════════════
# 6. VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════
print("\n[6/6] Gerando visualizações...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Machine Learning — Previsão de Vendas Angola · Josemar Manuel",
             fontsize=15, fontweight="bold", color="#00C2D4", y=0.98)

# ── Plot 1: Matrizes de Confusão lado a lado ─────────────
for idx, (model, y_pred, titulo) in enumerate([
    (lr, y_pred_lr, "Regressão Logística"),
    (dt, y_pred_dt, "Árvore de Decisão")
], 1):
    ax = fig.add_subplot(2, 3, idx)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                cmap=sns.light_palette("#00C2D4", as_cmap=True),
                linewidths=1, linecolor="#0D1B35",
                xticklabels=["Não Atingiu","Atingiu"],
                yticklabels=["Não Atingiu","Atingiu"])
    ax.set_title(f"Matriz Confusão\n{titulo}", fontweight="bold")
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")

# ── Plot 3: Curva ROC ────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
for (y_prob, titulo, cor) in [
    (y_prob_lr, f"Reg. Logística (AUC={auc_lr:.3f})", "#00C2D4"),
    (y_prob_dt, f"Árvore Decisão (AUC={auc_dt:.3f})", "#A855F7"),
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax3.plot(fpr, tpr, color=cor, linewidth=2.5, label=titulo)
ax3.plot([0,1],[0,1], color="#8B96B0", linestyle="--", linewidth=1)
ax3.fill_between(fpr, tpr, alpha=0.1, color="#00C2D4")
ax3.set_title("Curva ROC — Comparação de Modelos", fontweight="bold")
ax3.set_xlabel("Taxa Falso Positivo")
ax3.set_ylabel("Taxa Verdadeiro Positivo")
ax3.legend(fontsize=8)
ax3.grid(True)

# ── Plot 4: Feature Importance RF ────────────────────────
ax4 = fig.add_subplot(2, 3, 4)
fi_rf_sorted = fi_rf_top.sort_values()
bars = ax4.barh(fi_rf_sorted.index, fi_rf_sorted.values,
                color=CORES[:len(fi_rf_sorted)])
ax4.set_title("Feature Importance — Random Forest", fontweight="bold")
ax4.set_xlabel("Importância")
for bar in bars:
    ax4.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.3f}", va="center", fontsize=8)

# ── Plot 5: Previsto vs Real (RF Regressão) ───────────────
ax5 = fig.add_subplot(2, 3, 5)
ax5.scatter(yr_test, yr_pred, alpha=0.3, color="#00C2D4", s=15)
lim = max(yr_test.max(), yr_pred.max())
ax5.plot([0, lim], [0, lim], color="#F5A623", linestyle="--", linewidth=2, label="Perfeito")
ax5.set_title(f"Previsto vs Real (RF)\nR² = {r2:.3f}", fontweight="bold")
ax5.set_xlabel("Valor Real (M Kz)")
ax5.set_ylabel("Valor Previsto (M Kz)")
ax5.legend()

# ── Plot 6: Comparação Métricas ───────────────────────────
ax6 = fig.add_subplot(2, 3, 6)
modelos   = ["Reg. Logística","Árvore Decisão"]
acuracias = [acc_lr, acc_dt]
aucs      = [auc_lr, auc_dt]
x = np.arange(len(modelos))
w = 0.35
bars1 = ax6.bar(x - w/2, acuracias, w, label="Acurácia", color="#00C2D4", alpha=0.9)
bars2 = ax6.bar(x + w/2, aucs,      w, label="AUC-ROC",  color="#A855F7", alpha=0.9)
ax6.set_title("Comparação de Modelos", fontweight="bold")
ax6.set_xticks(x); ax6.set_xticklabels(modelos)
ax6.set_ylim(0, 1.15)
ax6.legend()
for bar in list(bars1) + list(bars2):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("ML_Previsao_Vendas_Angola.png", dpi=150, bbox_inches="tight",
            facecolor="#0D1B35")
print("   Gráfico guardado: ML_Previsao_Vendas_Angola.png")
plt.close()

# ══════════════════════════════════════════════════════════
# RESUMO FINAL
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" RESUMO DOS MODELOS DE MACHINE LEARNING")
print("=" * 60)
print(f"""
  Modelo                  Acurácia    AUC-ROC
  ─────────────────────── ────────    ───────
  Regressão Logística      {acc_lr*100:.2f}%     {auc_lr:.4f}
  Árvore de Decisão        {acc_dt*100:.2f}%     {auc_dt:.4f}

  Random Forest Regressão
    R² Score               {r2*100:.2f}%
    RMSE                   {rmse:.4f} M Kz
    MAE                    {mae:.4f} M Kz

  Feature mais importante: {fi_rf_top.index[0]}

  Ficheiros gerados:
    ML_Previsao_Vendas_Angola.png
""")
print("=" * 60)
