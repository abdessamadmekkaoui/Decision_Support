#!/usr/bin/env python3
"""
Script de lancement rapide pour le Dashboard Prédiction des Ventes
Exécutez ce script pour démarrer l'application Streamlit
"""

import sys
import os
import subprocess
import pandas as pd
from pathlib import Path

def check_requirements():
    """Vérifier que tous les fichiers requis sont présents"""
    print("🔍 Vérification des prérequis...")
    
    # Vérifier les fichiers obligatoires
    required_files = ['app.py', 'requirements.txt']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Fichiers manquants: {', '.join(missing_files)}")
        return False
    
    # Vérifier le fichier de données
    if not os.path.exists('merged.csv'):
        print("⚠️  ATTENTION: fichier 'merged.csv' non trouvé")
        print("   → L'application utilisera des données de démonstration")
        print("   → Pour utiliser vos vraies données, placez 'merged.csv' dans ce dossier")
    else:
        print("✅ Fichier de données 'merged.csv' trouvé")
        # Vérifier la structure du CSV
        try:
            df = pd.read_csv('merged.csv', nrows=1)
            required_columns = ['Date', 'Branch', 'Sales Excl', 'PAX', 'Qty Sold']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️  Colonnes manquantes dans merged.csv: {', '.join(missing_cols)}")
                print("   → L'application créera des colonnes par défaut")
            else:
                print("✅ Structure du fichier CSV validée")
        except Exception as e:
            print(f"⚠️  Erreur lors de la lecture de merged.csv: {e}")
    
    # Vérifier les modèles de prédiction
    model_files = [
        'random_forest_model.pkl',
        'lightgbm_model.pkl', 
        'xgboost_model.pkl',
        'label_encoder.pkl'
    ]
    
    models_found = [f for f in model_files if os.path.exists(f)]
    if models_found:
        print(f"✅ Modèles IA trouvés: {', '.join(models_found)}")
    else:
        print("ℹ️  Aucun modèle .pkl trouvé")
        print("   → La page prédiction utilisera une simulation intelligente")
        print("   → Pour utiliser vos vrais modèles, placez vos fichiers .pkl ici")
    
    print("✅ Vérification terminée")
    return True

def install_dependencies():
    """Installer les dépendances requises"""
    print("📦 Installation des dépendances...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dépendances installées avec succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation des dépendances: {e}")
        return False

def launch_streamlit():
    """Lancer l'application Streamlit"""
    print("🚀 Lancement du dashboard...")
    print("📊 L'application va s'ouvrir dans votre navigateur...")
    print("🌐 URL: http://localhost:8501")
    print("\n" + "="*50)
    print("DASHBOARD PRÉDICTION DES VENTES")
    print("="*50)
    print("Pour arrêter l'application: Ctrl+C")
    print("="*50 + "\n")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\n🛑 Application arrêtée par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")

def main():
    """Fonction principale"""
    print("🏪 DASHBOARD PRÉDICTION DES VENTES")
    print("=" * 40)
    
    # Vérifier les prérequis
    if not check_requirements():
        print("❌ Prérequis non satisfaits. Arrêt du script.")
        sys.exit(1)
    
    print("\n" + "-" * 40)
    
    # Demander si installer les dépendances
    install_deps = input("📦 Installer/mettre à jour les dépendances? (y/N): ").lower().strip()
    if install_deps in ['y', 'yes', 'o', 'oui']:
        if not install_dependencies():
            print("❌ Impossible d'installer les dépendances. Arrêt du script.")
            sys.exit(1)
    
    print("\n" + "-" * 40)
    
    # Lancer l'application
    launch_streamlit()

if __name__ == "__main__":
    main()