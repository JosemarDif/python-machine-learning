# 🤖 Mineração de Dados & Machine Learning — Python

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Concluído-4ADE80?style=flat)
![Modelos](https://img.shields.io/badge/Modelos-3%20Algoritmos-A855F7?style=flat)

> Implementação de 3 modelos de Machine Learning para classificação e regressão de dados de vendas angolanas: Regressão Logística, Árvore de Decisão e Random Forest.

---

## 🎯 Problema de Negócio

| Problema | Tipo ML | Modelo |
|----------|---------|--------|
| "Esta venda vai atingir a meta?" | Classificação Binária | Regressão Logística + Árvore de Decisão |
| "Qual o valor previsto desta venda?" | Regressão | Random Forest |

---

## 📊 Resultados dos Modelos

| Modelo | Acurácia | AUC-ROC | Técnica |
|--------|----------|---------|---------|
| Regressão Logística | ~72% | ~0.78 | Classificação Linear |
| Árvore de Decisão (depth=5) | ~74% | ~0.80 | CART |
| Random Forest (Regressão) | R²~0.85 | — | Ensemble |

---

## 🛠️ Instalação e Execução

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python machine_learning_vendas.py
```

---

## 🔬 Pipeline Completo

```
Dados Brutos
    │
    ▼
Feature Engineering
 • Codificação com LabelEncoder (Vendedor, Região, Produto)
 • Criação de variáveis derivadas (Trimestre, Lucro)
    │
    ▼
Pré-processamento
 • Train/Test Split (80/20, stratificado)
 • StandardScaler (para Regressão Logística)
    │
    ├──► Regressão Logística → Avaliação (Acc, AUC, Report)
    ├──► Árvore de Decisão   → Avaliação + Feature Importance
    └──► Random Forest       → Avaliação (R², RMSE, MAE)
    │
    ▼
Cross-Validation (5-fold StratifiedKFold)
    │
    ▼
Visualizações
 • Matrizes de Confusão
 • Curva ROC comparativa
 • Feature Importance
 • Previsto vs Real
 • Comparação de métricas
```

---

## 📐 Medidas de Avaliação Usadas

**Classificação:**
- **Accuracy** — % de previsões corretas
- **AUC-ROC** — Área sob a curva ROC (quanto maior, melhor)
- **Precision / Recall / F1** — Para classes desbalanceadas
- **Matriz de Confusão** — TP, FP, TN, FN

**Regressão:**
- **R²** — Coeficiente de determinação (% variância explicada)
- **RMSE** — Root Mean Squared Error (penaliza grandes erros)
- **MAE** — Mean Absolute Error (erro médio absoluto)

---

## 💡 Feature Importance (Top Features)

As variáveis mais importantes para prever o valor de venda:
1. `Preço_Unit` — Preço unitário do produto
2. `Quantidade` — Número de unidades vendidas
3. `Produto_enc` — Tipo de produto
4. `Desconto_Pct` — Desconto aplicado
5. `Experiência_Anos` — Experiência do vendedor

---

## 📁 Estrutura

```
Python_ML/
├── README.md
└── machine_learning_vendas.py   ← Script completo auto-suficiente
```

---

## 🏷️ Tecnologias

`Python` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Regressão Logística` `Árvore de Decisão` `Random Forest` `Cross-Validation` `ROC Curve` `Feature Importance` `StandardScaler` `LabelEncoder`

---

*Autor: **Josemar Manuel** · [LinkedIn](https://linkedin.com/in/josemarmanuel) · Luanda, Angola*
