import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier   # exige xgboost instalado


# Carregamento do dataset
caminho_arquivo = "Dados/cardio_data_processed.csv"
df = pd.read_csv(caminho_arquivo)

print("Formato do dataset:", df.shape)
print(df.head())


# Definição de X e y
y = df["cardio"]
colunas_para_dropar = ["id", "age", "bp_category", "bp_category_encoded"]
X = df.drop(columns=colunas_para_dropar + ["cardio"])

print("\nColunas utilizadas como preditoras:")
print(X.columns.tolist())


# Separação treino/teste e padronização
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Modelos utilizados
modelos = {
    "DecisionTree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ),
    "XGBoost": XGBClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )
}


# Treinamento, avaliação e tabela de resultados
resultados = []

for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    y_prob = modelo.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    resultados.append({
        "modelo": nome,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "roc_auc": auc
    })

df_resultados = pd.DataFrame(resultados)
print("\nTabela de métricas por modelo:")
print(df_resultados)


# Exibição dos resultados de cada modelo em português
for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_test_scaled)
    y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 60)
    print(f"Resultados do modelo: {nome}")
    print("=" * 60)
    print("Matriz de confusão (linhas = classe real, colunas = previsão):")
    print(cm)
    print(f"\nVerdadeiros negativos (classe 0 prevista corretamente): {tn}")
    print(f"Falsos positivos  (classe 0 prevista como 1): {fp}")
    print(f"Falsos negativos  (classe 1 prevista como 0): {fn}")
    print(f"Verdadeiros positivos (classe 1 prevista corretamente): {tp}\n")

    print(f"Acurácia : {acc:.3f}  -> proporção total de acertos.")
    print(f"Precisão : {prec:.3f}  -> entre os previstos como 'doente', "
          f"quantos realmente eram doentes.")
    print(f"Recall   : {rec:.3f}  -> entre todos os doentes reais, "
          f"quantos o modelo identificou (sensibilidade).")
    print(f"ROC AUC  : {auc:.3f}  -> capacidade global de separar "
          f"pacientes doentes e não doentes.\n")


# Seleção do melhor modelo pela métrica ROC AUC
melhor_idx = df_resultados["roc_auc"].idxmax()
melhor_nome = df_resultados.loc[melhor_idx, "modelo"]
melhor_modelo = modelos[melhor_nome]

print("=" * 60)
print(f"\nMelhor modelo (pela ROC AUC): {melhor_nome}")
print("=" * 60)

y_pred_best = melhor_modelo.predict(X_test_scaled)
cm_best = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm_best.ravel()

print("\nMatriz de confusão do melhor modelo:")
print(cm_best)
print(f"\nVerdadeiros negativos: {tn}")
print(f"Falsos positivos     : {fp}")
print(f"Falsos negativos     : {fn}")
print(f"Verdadeiros positivos: {tp}")


# Resumo automático dos resultados do melhor modelo
melhor_linha = df_resultados.loc[melhor_idx]

texto_resumo = f"""
Resumo automático dos resultados do melhor modelo:

O modelo com melhor desempenho foi o {melhor_linha['modelo']},
com acurácia de {melhor_linha['accuracy']:.3f},
precisão de {melhor_linha['precision']:.3f},
recall (sensibilidade) de {melhor_linha['recall']:.3f}
e ROC AUC de {melhor_linha['roc_auc']:.3f} no conjunto de teste.

Essas métricas indicam a qualidade do modelo em identificar corretamente
pacientes com e sem risco de doença cardiovascular.
"""

print(texto_resumo)


# Texto comparativo entre DecisionTree e XGBoost
arvore = df_resultados[df_resultados["modelo"] == "DecisionTree"].iloc[0]
xgb    = df_resultados[df_resultados["modelo"] == "XGBoost"].iloc[0]

texto_comparativo = f"""
Comparação entre os modelos:

DecisionTree:
    acurácia = {arvore['accuracy']:.3f},
    precisão = {arvore['precision']:.3f},
    recall = {arvore['recall']:.3f},
    ROC AUC = {arvore['roc_auc']:.3f}.

XGBoost:
    acurácia = {xgb['accuracy']:.3f},
    precisão = {xgb['precision']:.3f},
    recall = {xgb['recall']:.3f},
    ROC AUC = {xgb['roc_auc']:.3f}.

De forma geral, o XGBoost apresenta desempenho superior, especialmente
em recall e ROC AUC, sugerindo maior capacidade de identificar pacientes
com risco de doença cardiovascular e melhor discriminação global entre
doentes e não doentes quando comparado à árvore de decisão.
"""

print(texto_comparativo)
