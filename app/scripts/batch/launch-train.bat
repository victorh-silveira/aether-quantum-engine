@echo off
TITLE Aether Engine - Treino Deep Learning + Meta-Classificador

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

if "%CONDA_ACTIVATE%"=="" (
    echo [ERRO] Nao foi possivel localizar o activate.bat do Anaconda.
    pause
    exit /b 1
)

echo [AETHER] Iniciando treino Deep Learning e meta-classificador...
cd /d "%REPO_ROOT%"
start "AETHER TRAIN DL" cmd /k ""%~dp0_run_train.bat" "%CONDA_ACTIVATE%""
timeout /t 2 /nobreak > nul
start "AETHER TRAIN META" cmd /k ""%~dp0_run_meta_train.bat" "%CONDA_ACTIVATE%""
echo [OK] Treino DL e meta-classificador em execucao.
timeout /t 3 /nobreak > nul
exit /b 0
