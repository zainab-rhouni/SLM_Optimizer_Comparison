# 🔬 Comparaison des Algorithmes d'Optimisation pour le Fine-tuning d'un Small Language Model (SLM)

Ce projet présente une comparaison expérimentale de **cinq algorithmes d'optimisation** utilisés lors du fine-tuning d'un **Small Language Model (SLM)** sur une tâche de classification de sentiments.

L'objectif est d'analyser l'impact des optimiseurs sur :
- la performance du modèle
- la convergence
- la consommation mémoire GPU
- le temps d'entraînement
- la stabilité et la généralisation

---

# 🧠 Modèle et Tâche

## Modèle utilisé
- **Architecture :** DistilBERT-base-uncased
- **Nombre de paramètres :** 66M

## Tâche
Classification binaire de sentiments sur :

- **Dataset :** SST-2 (Stanford Sentiment Treebank)
- **Classes :**
  - Positive
  - Negative

## Données utilisées

| Ensemble | Nombre d'échantillons |
|----------|----------------------|
| Train | 67 349 |
| Validation | 872 |
| Test | 1 821 |

## Métriques évaluées

- Accuracy
- F1-Score
- Loss
- Temps d'entraînement
- Mémoire GPU utilisée
- Stabilité de convergence

---

# 🚀 Optimiseurs Comparés

| Optimiseur | Configuration |
|------------|--------------|
| AdamW | lr = 2e-5, weight_decay = 0.01 |
| SGD | lr = 1e-3, momentum = 0.9, weight_decay = 0.01 |
| Adagrad | lr = 1e-3, weight_decay = 0.01 |
| RMSprop | lr = 1e-4, weight_decay = 0.01 |
| AdaFactor | relative_step=True, scale_parameter=True |

---

# 📊 Résultats Expérimentaux

## Performances finales

| Optimiseur | Accuracy | F1-Score | Temps (s) | Mémoire GPU (MB) |
|------------|----------|----------|-----------|------------------|
| RMSprop | 0.864 | 0.861 | 62.8 | 1918.3 |
| Adagrad | 0.858 | 0.859 | 62.9 | 1906.1 |
| AdamW | 0.840 | 0.850 | 58.2 | 2197.8 |
| AdaFactor | 0.850 | 0.847 | 71.3 | 1639.3 |
| SGD | 0.804 | 0.819 | 60.9 | 1917.9 |

---

# 📌 Analyse des résultats

### 🥇 RMSprop
- Meilleure Accuracy et meilleur F1-score
- Bon équilibre entre performance et consommation mémoire

### 🥈 Adagrad
- Très proche de RMSprop
- Bonne stabilité pendant l'entraînement

### ⚡ AdamW
- Temps d'entraînement le plus faible
- Consommation mémoire GPU plus importante

### 💾 AdaFactor
- Optimiseur le plus économique en mémoire
- Adapté aux environnements avec ressources limitées

### 📉 SGD
- Performance plus faible
- Mais comportement stable pendant l'apprentissage

---

# 📈 Visualisations Générées

Le notebook produit plusieurs analyses graphiques :

## 1. Convergence des optimiseurs
- Evolution de la Loss
- Evolution de l'Accuracy
- Comparaison Train / Validation

## 2. Métriques finales
- Accuracy finale
- F1-score
- Temps d'entraînement

## 3. Analyse mémoire GPU
- Consommation mémoire par époque
- Compromis mémoire / performance

## 4. Evolution du F1-score
- Progression du modèle durant les epochs

## 5. Radar Chart
Comparaison multi-critères :

- Accuracy
- F1-score
- Vitesse
- Mémoire

## 6. Sensibilité au Learning Rate
Impact du learning rate sur :
- AdamW
- SGD

## 7. Matrices de confusion
Analyse des erreurs de classification.

## 8. Gap de généralisation
Comparaison entre :
- Train Accuracy
- Validation Accuracy

---

# 🛠️ Installation

Installer les dépendances :

```bash
pip install transformers datasets accelerate evaluate

pip install torch torchvision torchaudio

pip install matplotlib seaborn scikit-learn pandas numpy

pip install optax
