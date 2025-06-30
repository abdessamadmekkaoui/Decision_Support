@echo off
echo.
echo ==========================================
echo   DASHBOARD PREDICTION DES VENTES
echo ==========================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERREUR: Python n'est pas installe ou pas dans le PATH
    echo Veuillez installer Python depuis https://python.org
    pause
    exit /b 1
)

echo Python detecte: 
python --version

echo.
echo Installation des dependances...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERREUR: Impossible d'installer les dependances
    echo Verifiez votre connexion internet et les permissions
    pause
    exit /b 1
)

echo.
echo ==========================================
echo   LANCEMENT DU DASHBOARD
echo ==========================================
echo.
echo Le dashboard va s'ouvrir dans votre navigateur...
echo URL: http://localhost:8501
echo.
echo Pour arreter: Fermez cette fenetre ou Ctrl+C
echo.

REM Lancer Streamlit
streamlit run app.py

echo.
echo Dashboard ferme.
pause