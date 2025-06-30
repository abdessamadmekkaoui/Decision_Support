# Configuration pour l'application Dashboard Prédiction des Ventes

# Chemins des fichiers
DATA_PATH = "merged.csv"  # Remplacez par le chemin vers vos données
MODEL_PATHS = {
    'random_forest': 'random_forest_model.pkl',
    'lightgbm': 'lightgbm_model.pkl',
    'xgboost': 'xgboost_model.pkl',
    'label_encoder': 'label_encoder.pkl'
}

# Configuration des colonnes (adaptez selon votre dataset)
COLUMN_MAPPING = {
    'date': 'Date',
    'branch': 'Branch',
    'sales_excl': 'Sales Excl',
    'sales_tax': 'Sales Tax',
    'sales_incl': 'Sales Incl',
    'pax': 'PAX',
    'qty_sold': 'Qty Sold',
    'category': 'Category',  # Optionnel - à ajouter si disponible
    'department': 'Major_Department_Name',  # Optionnel - à ajouter si disponible
    'avg_basket_value': 'Avg Basket Value',
    'avg_basket_value_excl': 'Avg Basket Value Excl Vat',
    'avg_basket_qty': 'Avg Basket Qty',
    'avg_item_value_incl': 'Avg Item Value Incl',
    'avg_item_value_excl': 'Avg Item Value Excl'
}

# Configuration des métriques
METRICS_CONFIG = {
    'gross_profit_margin': 0.3,  # 30% de marge par défaut
    'currency': 'MAD',
    'date_format': '%Y-%m-%d'
}

# Configuration des graphiques
CHART_CONFIG = {
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'theme': 'plotly_white',
    'height_default': 400,
    'height_large': 600
}

# Configuration de la prédiction
PREDICTION_CONFIG = {
    'features': [
        'Branch', 'Year', 'Month', 'Day', 
        'DayOfWeek', 'WeekOfYear', 'Quarter', 'IsWeekend', 
        'IsMonthStart', 'IsMonthEnd', 'IsHoliday', 'IsRamadan', 
        'IsEid', 'DaysSinceStart', 'DayOfMonth', 'IsSchoolVacation', 
        'Temperature', 'Precipitation', 'ExtremeWeather', 'IsPayday'
    ],
    'target': 'Sales Excl',
    'training_start_date': '2023-01-01'
}

# Configuration de l'interface utilisateur
UI_CONFIG = {
    'page_title': "Dashboard Prédiction des Ventes",
    'page_icon': "📊",
    'layout': "wide",
    'sidebar_state': "expanded"
}

# Messages et textes
MESSAGES = {
    'welcome': "Bienvenue sur votre Dashboard de Prédiction des Ventes",
    'prediction_success': "✅ Prédiction réalisée avec succès!",
    'error_model_loading': "❌ Erreur lors du chargement des modèles",
    'error_data_loading': "❌ Erreur lors du chargement des données"
}

# Validation des données
DATA_VALIDATION = {
    'required_columns': ['Date', 'Branch', 'Sales Excl', 'PAX', 'Qty Sold'],
    'date_range': {
        'min_year': 2023,
        'max_year': 2025
    },
    'numeric_columns': ['Sales Excl', 'Sales Tax', 'Sales Incl', 'PAX', 'Qty Sold']
}