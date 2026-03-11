# Water Pump Challenge

Exercice de Machine Learning : prédire l'état de fonctionnement des pompes à eau en Tanzanie.

## Objectif

Construire un modèle capable de prédire si une pompe à eau est :
- `functional` : opérationnelle
- `functional needs repair` : fonctionne mais nécessite des réparations
- `non functional` : non opérationnelle

## Données

```
data/
├── train_features.csv    # Features d'entraînement (47 520 échantillons)
├── train_labels.csv      # Labels d'entraînement
├── test_features.csv     # Features de test (11 880 échantillons)
├── test_labels_secret.csv # Labels de test (pour évaluation)
└── sample_submission.csv  # Exemple de format de soumission
```

## Pour commencer

1. Ouvrir `getting_started.ipynb`
2. Créer votre modèle et générer `ma_soumission.csv`
3. Évaluer avec `evaluate_submission.ipynb`

## Format de soumission

```csv
id,status_group
69572,functional
8776,non functional
...
```

## Métrique

Accuracy (taux de classification correcte)
