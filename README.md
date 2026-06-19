Comparaison des Algorithmes d'Optimisation pour le Fine-tuning de SLM
Ce projet compare les performances de cinq optimiseurs populaires pour le fine-tuning d'un Small Language Model (SLM) sur une tâche de classification de sentiments.

🧠 Modèle et Tâche
Modèle : DistilBERT-base-uncased (66M paramètres)

Tâche : Classification de sentiments sur le dataset SST-2 (Stanford Sentiment Treebank)

Dataset : 67 349 échantillons d'entraînement, 872 de validation, 1 821 de test

Métriques : Loss, Accuracy, F1-Score, Mémoire GPU, Temps d'entraînement, Stabilité

🚀 Optimiseurs Comparés
Optimiseur	Configuration
AdamW	lr=2e-5, weight_decay=0.01
SGD	lr=1e-3, momentum=0.9, weight_decay=0.01
Adagrad	lr=1e-3, weight_decay=0.01
RMSprop	lr=1e-4, weight_decay=0.01
AdaFactor	relative_step=True, scale_parameter=True
📊 Résultats Clés
Optimiseur	Accuracy	F1-Score	Temps (s)	Mémoire (MB)
RMSprop	0.864	0.861	62.8	1918.3
Adagrad	0.858	0.859	62.9	1906.1
AdamW	0.840	0.850	58.2	2197.8
AdaFactor	0.850	0.847	71.3	1639.3
SGD	0.804	0.819	60.9	1917.9
Observations principales :
RMSprop offre le meilleur compromis performance/mémoire

Adagrad est très proche avec une excellente stabilité

AdamW est le plus rapide mais consomme plus de mémoire

AdaFactor est le plus économe en mémoire mais plus lent

SGD est le moins performant mais très stable

📈 Visualisations Générées
Le notebook produit 8 figures d'analyse :

Convergence des optimiseurs : Évolution de la loss et de l'accuracy (train/val)

Métriques finales : Accuracy, F1-Score et temps d'entraînement

Analyse mémoire GPU : Consommation par époque et compromis mémoire/F1

Évolution du F1-Score : Progression du F1 au cours des époques

Radar Chart : Comparaison multi-critères (Accuracy, F1, Vitesse, Mémoire)

Sensibilité au learning rate : Impact du LR sur AdamW et SGD

Matrices de confusion : Visualisation des erreurs de classification

Gap de généralisation : Écart entre train accuracy et val accuracy

🛠️ Installation et Exécution
Dépendances
bash
pip install transformers datasets accelerate evaluate
pip install torch torchvision torchaudio
pip install matplotlib seaborn scikit-learn pandas numpy
pip install optax  # Optionnel pour AdaFactor
Exécution
Ouvrir le notebook dans Google Colab ou Jupyter

Exécuter les cellules séquentiellement

Les figures sont sauvegardées automatiquement en PNG

📂 Structure du Code
text
📁 SLM_Optimizer_Comparison
├── 📓 SLM_Optimizer_Comparison.ipynb  # Notebook principal
├── 📊 fig1_convergence.png
├── 📊 fig2_final_metrics.png
├── 📊 fig3_memory.png
├── 📊 fig4_f1_progress.png
├── 📊 fig5_radar.png
├── 📊 fig6_lr_sensitivity.png
├── 📊 fig7_confusion_matrices.png
└── 📊 fig8_generalization.png
🔬 Détails Techniques
Époques : 3

Batch size : 32

Max length : 128 tokens

Training samples : 2000 (accélération)

Validation samples : 500

Device : CUDA (GPU Tesla T4)

📝 Interprétation des Résultats
Pour un usage en production :
Prioriser RMSprop pour une bonne performance avec une mémoire modérée

Utiliser AdaFactor si la mémoire GPU est très limitée

Préférer AdamW si la vitesse est critique

Stabilité et généralisation :
AdaFactor et SGD montrent la meilleure stabilité (std loss faible)

Tous les optimiseurs présentent un bon gap de généralisation (< 5%)

🤝 Contribution
Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request pour améliorer ce benchmark.
