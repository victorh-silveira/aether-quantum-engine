@echo off
TITLE Aether Engine - Treino Rise-Fall TCN + Meta
set "PYTHONASYNCIODEBUG="
set "PYTHONDEVMODE="
set "PYTHONUNBUFFERED=1"

pushd "%~dp0..\..\.."
SET "REPO_ROOT=%CD%"
popd
SET "ENV_NAME=deriv-api"

set "CONDA_ACTIVATE="
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    set "CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat"
) else if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    set "CONDA_ACTIVATE=C:\ProgramData\anaconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"
)

if not "%CONDA_ACTIVATE%"=="" (
    call "%CONDA_ACTIVATE%" %ENV_NAME%
)

set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\%ENV_NAME%\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\%ENV_NAME%\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

cd /d "%REPO_ROOT%"
echo [AETHER] launch-train Rise/Fall - TCN M5 + meta LightGBM
"%PYTHON_EXE%" -u app/scripts/operations/sanitize_fresh_run.py
if errorlevel 1 exit /b 1

pushd "%REPO_ROOT%\app"
"%PYTHON_EXE%" -u -m scripts.operations.train_loss_classifier
if errorlevel 1 (
    popd
    exit /b 1
)
popd

"%PYTHON_EXE%" -u app/train.py
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" -u app/scripts/operations/check_dl_checkpoint.py
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" -u app/scripts/operations/ensure_timescale.py
if errorlevel 1 echo [AVISO] Timescale seed indisponivel; meta tentara API Deriv.

"%PYTHON_EXE%" -u app/scripts/operations/train_meta_classifier.py --trials 60 --bars 5000 --source auto --candidate-on-low-quality --export-min-zscore -0.05 --export-min-ir -0.50
if errorlevel 1 exit /b 1

echo [AETHER] Treino concluido; checkpoint TCN compativel sob teto de 1%% da banca.
exit /b 0
