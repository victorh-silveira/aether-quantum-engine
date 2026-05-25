@echo off
TITLE Aether Engine - Master Launcher (LIVE)

:: Configurações de Ambiente
:: Configurações de Ambiente (Detecta a raiz automaticamente)
pushd "%~dp0..\.."
SET "APP_DIR=%CD%"
popd
SET "ENV_NAME=deriv-api"

:: Tenta localizar o script de ativação do Anaconda automaticamente
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

echo [AETHER] Iniciando Infraestrutura Aether Engine (MODO LIVE)...
cd /d "%APP_DIR%"

:: 1. Inicia o Monitor Primeiro (Visao Geral)
start "AETHER MONITOR" cmd /k "call "%CONDA_ACTIVATE%" %ENV_NAME% && python -m scripts.monitor.live_monitor"
timeout /t 3 /nobreak > nul

:: 2. Inicia o bot (modo LIVE via trading.mode em config/settings.json e AETHER_LIVE_TOKEN no .env)
start "AETHER LIVE" cmd /k "call "%CONDA_ACTIVATE%" %ENV_NAME% && python run.py"

echo [OK] Robotica e Monitor em execucao.
timeout /t 3 /nobreak > nul
exit /b 0
