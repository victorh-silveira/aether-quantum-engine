@echo off
TITLE Aether Engine - Treino Deep Learning + Meta-Classificador
set "PYTHONASYNCIODEBUG="
set "PYTHONDEVMODE="

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

echo [AETHER] Etapa 1/2: Executando treino Deep Learning (TCN)...
call "%~dp0_run_train.bat" "%CONDA_ACTIVATE%"
if errorlevel 1 (
    echo.
    echo [ERRO] Treino DL falhou. Treino do Meta-Classificador abortado para evitar race condition.
    pause
    exit /b 1
)

echo.
echo [AETHER] Etapa 2/2: Treino DL concluido com sucesso! Iniciando Meta-Classificador (LightGBM)...
call "%~dp0_run_meta_train.bat" "%CONDA_ACTIVATE%"
if errorlevel 1 (
    echo.
    echo [ERRO] Treino do Meta-Classificador falhou.
    pause
    exit /b 1
)

echo.
echo [SUCESSO] Pipeline de treinamento AETHER concluido com sucesso!
timeout /t 5 /nobreak > nul
exit /b 0
