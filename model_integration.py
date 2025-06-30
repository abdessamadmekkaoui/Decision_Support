"""
Script d'intégration des modèles de prédiction dans l'application Streamlit
Utilisez ce script pour adapter vos modèles existants à l'application
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

class ModelIntegrator:
    """
    Classe pour intégrer vos modèles de prédiction dans l'application Streamlit
    """
    
    def __init__(self, model_paths=None):
        """
        Initialiser l'intégrateur de modèles
        
        Args:
            model_paths (dict): Dictionnaire avec les chemins vers vos modèles
                Exemple: {
                    'random_forest': 'path/to/rf_model.pkl',
                    'lightgbm': 'path/to/lgb_model.pkl',
                    'xgboost': 'path/to/xgb_model.pkl',
                    'label_encoder': 'path/to/encoder.pkl'
                }
        """
        self.model_paths = model_paths or {}
        self.models = {}
        self.label_encoder = None
        
    def load_models(self):
        """
        Charger tous les modèles depuis les fichiers pickle
        """
        try:
            if 'random_forest' in self.model_paths:
                self.models['random_forest'] = joblib.load(self.model_paths['random_forest'])
                print("✅ Modèle Random Forest chargé")
                
            if 'lightgbm' in self.model_paths:
                self.models['lightgbm'] = joblib.load(self.model_paths['lightgbm'])
                print("✅ Modèle LightGBM chargé")
                
            if 'xgboost' in self.model_paths:
                self.models['xgboost'] = joblib.load(self.model_paths['xgboost'])
                print("✅ Modèle XGBoost chargé")
                
            if 'label_encoder' in self.model_paths:
                self.label_encoder = joblib.load(self.model_paths['label_encoder'])
                print("✅ Label Encoder chargé")
                
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles: {e}")
            return False
    
    def prepare_prediction_data(self, branch, date, temperature, precipitation, 
                             is_holiday=0, is_ramadan=0, is_eid=0, is_school_vacation=0):
        """
        Préparer les données pour la prédiction selon le format de vos modèles
        
        Args:
            branch (str): Nom de la branche
            date (str): Date au format 'YYYY-MM-DD'
            temperature (float): Température en Celsius
            precipitation (float): Précipitations en mm
            is_holiday (int): 1 si jour férié, 0 sinon
            is_ramadan (int): 1 si Ramadan, 0 sinon
            is_eid (int): 1 si Eid, 0 sinon
            is_school_vacation (int): 1 si vacances scolaires, 0 sinon
            
        Returns:
            pd.DataFrame: Données formatées pour la prédiction
        """
        try:
            # Convertir la date
            date_dt = datetime.strptime(date, '%Y-%m-%d')
            
            # Calculer les features temporelles
            day_of_week = date_dt.weekday()
            week_of_year = date_dt.isocalendar()[1]
            quarter = (date_dt.month - 1) // 3 + 1
            is_weekend = 1 if day_of_week >= 5 else 0
            is_month_start = 1 if date_dt.day == 1 else 0
            
            # Vérifier si c'est la fin du mois
            next_day = date_dt + pd.Timedelta(days=1)
            is_month_end = 1 if next_day.month != date_dt.month else 0
            
            # Calculer les jours depuis le début de l'entraînement
            training_start = datetime(2023, 1, 1)  # Adaptez selon vos données
            days_since_start = (date_dt - training_start).days
            
            # Jour de paie (15 ou fin de mois)
            is_payday = 1 if (date_dt.day == 15) or is_month_end else 0
            
            # Météo extrême
            extreme_weather = 1 if (temperature > 30) or (temperature < 0) or (precipitation > 10) else 0
            
            # Créer le DataFrame avec toutes les features
            prediction_data = pd.DataFrame({
                'Branch': [branch],
                'Year': [date_dt.year],
                'Month': [date_dt.month],
                'Day': [date_dt.day],
                'DayOfWeek': [day_of_week],
                'WeekOfYear': [week_of_year],
                'Quarter': [quarter],
                'IsWeekend': [is_weekend],
                'IsMonthStart': [is_month_start],
                'IsMonthEnd': [is_month_end],
                'IsHoliday': [is_holiday],
                'IsRamadan': [is_ramadan],
                'IsEid': [is_eid],
                'DaysSinceStart': [days_since_start],
                'DayOfMonth': [date_dt.day],
                'IsSchoolVacation': [is_school_vacation],
                'Temperature': [temperature],
                'Precipitation': [precipitation],
                'ExtremeWeather': [extreme_weather],
                'IsPayday': [is_payday]
            })
            
            # Encoder la branche si l'encodeur est disponible
            if self.label_encoder:
                try:
                    prediction_data['Branch'] = self.label_encoder.transform([branch])
                except ValueError as e:
                    print(f"⚠️ Attention: Branche '{branch}' non reconnue par l'encodeur")
                    # Utiliser la première classe connue par défaut
                    prediction_data['Branch'] = [0]
            
            return prediction_data
            
        except Exception as e:
            print(f"❌ Erreur lors de la préparation des données: {e}")
            return None
    
    def predict(self, branch, date, temperature, precipitation, 
               is_holiday=0, is_ramadan=0, is_eid=0, is_school_vacation=0):
        """
        Faire des prédictions avec tous les modèles disponibles
        
        Returns:
            dict: Dictionnaire avec les prédictions de chaque modèle
        """
        try:
            # Préparer les données
            prediction_data = self.prepare_prediction_data(
                branch, date, temperature, precipitation,
                is_holiday, is_ramadan, is_eid, is_school_vacation
            )
            
            if prediction_data is None:
                return None
            
            predictions = {}
            
            # Random Forest
            if 'random_forest' in self.models:
                rf_pred = self.models['random_forest'].predict(prediction_data)[0]
                predictions['Random Forest'] = round(float(rf_pred), 2)
            
            # LightGBM
            if 'lightgbm' in self.models:
                lgb_pred = self.models['lightgbm'].predict(prediction_data)[0]
                predictions['LightGBM'] = round(float(lgb_pred), 2)
            
            # XGBoost
            if 'xgboost' in self.models:
                xgb_pred = self.models['xgboost'].predict(prediction_data)[0]
                predictions['XGBoost'] = round(float(xgb_pred), 2)
            
            # Calculer la moyenne
            if predictions:
                avg_pred = sum(predictions.values()) / len(predictions)
                predictions['Average'] = round(avg_pred, 2)
            
            return predictions
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction: {e}")
            return None
    
    def validate_models(self, test_data_path=None):
        """
        Valider que les modèles fonctionnent correctement
        
        Args:
            test_data_path (str): Chemin vers un fichier de test (optionnel)
        """
        print("🔍 Validation des modèles...")
        
        # Test avec des données d'exemple
        test_prediction = self.predict(
            branch='Bir Jdid',  # Adaptez selon vos branches
            date='2025-06-01',
            temperature=20.0,
            precipitation=0.0,
            is_holiday=0,
            is_ramadan=0,
            is_eid=0,
            is_school_vacation=0
        )
        
        if test_prediction:
            print("✅ Test de prédiction réussi:")
            for model, pred in test_prediction.items():
                print(f"  - {model}: {pred}")
        else:
            print("❌ Échec du test de prédiction")
        
        return test_prediction is not None

# Fonction d'aide pour l'intégration dans Streamlit
@st.cache_resource
def load_prediction_models():
    """
    Fonction pour charger les modèles dans Streamlit avec cache
    Utilisez cette fonction dans votre app.py
    """
    model_paths = {
        'random_forest': 'random_forest_model.pkl',
        'lightgbm': 'lightgbm_model.pkl',
        'xgboost': 'xgboost_model.pkl',
        'label_encoder': 'label_encoder.pkl'
    }
    
    integrator = ModelIntegrator(model_paths)
    
    if integrator.load_models():
        return integrator
    else:
        st.error("❌ Impossible de charger les modèles de prédiction")
        return None

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration des chemins vers vos modèles
    model_paths = {
        'random_forest': 'random_forest_model.pkl',
        'lightgbm': 'lightgbm_model.pkl',
        'xgboost': 'xgboost_model.pkl',
        'label_encoder': 'label_encoder.pkl'
    }
    
    # Créer l'intégrateur
    integrator = ModelIntegrator(model_paths)
    
    # Charger les modèles
    if integrator.load_models():
        print("🎉 Tous les modèles ont été chargés avec succès!")
        
        # Valider les modèles
        if integrator.validate_models():
            print("🎯 Les modèles sont prêts pour l'intégration dans Streamlit!")
        else:
            print("⚠️ Attention: Problème détecté avec les modèles")
    else:
        print("❌ Impossible de charger les modèles")
        print("Vérifiez que les fichiers .pkl existent et sont dans le bon répertoire")