"""
Script pour préparer les données du Water Pump Challenge.
Crée un split train/test et génère les fichiers nécessaires.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Charger les données
data = pd.read_csv('data.csv')
target = pd.read_csv('target.csv')

# Fusionner pour assurer l'alignement
df = data.merge(target, on='id')

print(f"Nombre total d'échantillons: {len(df)}")
print(f"\nDistribution des classes:")
print(df['status_group'].value_counts())
print(f"\nPourcentages:")
print(df['status_group'].value_counts(normalize=True) * 100)

# Split stratifié 80% train / 20% test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['status_group']
)

print(f"\nTaille du jeu d'entraînement: {len(train_df)}")
print(f"Taille du jeu de test: {len(test_df)}")

# Créer le dossier data
os.makedirs('data', exist_ok=True)

# Séparer features et labels
feature_cols = [col for col in data.columns]
label_cols = ['id', 'status_group']

# Sauvegarder les fichiers d'entraînement
train_features = train_df[feature_cols]
train_labels = train_df[label_cols]
train_features.to_csv('data/train_features.csv', index=False)
train_labels.to_csv('data/train_labels.csv', index=False)

# Sauvegarder les fichiers de test (features publiques, labels secrets)
test_features = test_df[feature_cols]
test_labels = test_df[label_cols]
test_features.to_csv('data/test_features.csv', index=False)
test_labels.to_csv('data/test_labels_secret.csv', index=False)  # À ne pas partager!

# Créer un fichier exemple de soumission
sample_submission = test_labels.copy()
sample_submission['status_group'] = 'functional'  # Prédiction naïve
sample_submission.to_csv('data/sample_submission.csv', index=False)

print("\nFichiers créés dans le dossier 'data/':")
print("  - train_features.csv  (features d'entraînement)")
print("  - train_labels.csv    (labels d'entraînement)")
print("  - test_features.csv   (features de test - à prédire)")
print("  - test_labels_secret.csv (labels de test - NE PAS PARTAGER)")
print("  - sample_submission.csv  (exemple de format de soumission)")

# Vérification de la distribution dans train et test
print("\nDistribution dans train:")
print(train_labels['status_group'].value_counts(normalize=True) * 100)
print("\nDistribution dans test:")
print(test_labels['status_group'].value_counts(normalize=True) * 100)
