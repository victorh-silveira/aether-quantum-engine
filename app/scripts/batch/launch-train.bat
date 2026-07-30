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

echo(
echo [AETHER] Etapa 2/3: Verificando infraestrutura Docker (TimescaleDB)...
docker container inspect aether-timescaledb 1>nul 2>nul
if not errorlevel 1 (
    echo [AETHER] TimescaleDB ja esta saudavel.
    goto :meta_start
)
echo [AETHER] TimescaleDB offline - tentando subir stack core...
docker compose -f "%REPO_ROOT%\infra\docker\docker-compose.yml" --project-directory "%REPO_ROOT%\infra\docker" --env-file "%REPO_ROOT%\.env" up -d timescaledb 1>nul 2>nul
if errorlevel 1 (
    echo [AVISO] Docker Compose falhou. Meta-classificador usara API Deriv (fallback).
    goto :meta_start
)
echo [AETHER] Aguardando TimescaleDB ficar saudavel...
for /l %%i in (1,1,30) do (
    docker exec aether-timescaledb pg_isready -U aether 1>nul 2>nul
    if not errorlevel 1 goto :ts_ok
    timeout /t 2 /nobreak 1>nul
)
echo [AVISO] Timeout ao aguardar TimescaleDB. Meta-classificador usara fallback.
goto :meta_start

:ts_ok
echo [AETHER] TimescaleDB pronto.

:meta_start
echo.
echo [AETHER] Etapa 3/3: Treino DL concluido com sucesso! Iniciando Meta-Classificador (LightGBM)...
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
