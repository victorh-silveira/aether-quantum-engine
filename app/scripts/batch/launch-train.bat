@echo off
TITLE Aether Engine - Treino Deep Learning + Meta-Classificador
set PYTHONASYNCIODEBUG=
set PYTHONDEVMODE=

pushd "%~dp0..\..\.."
SET REPO_ROOT=%CD%
popd
SET ENV_NAME=deriv-api
set PYTHON_EXE=%USERPROFILE%\anaconda3\envs\%ENV_NAME%\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=%USERPROFILE%\miniconda3\envs\%ENV_NAME%\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

set CONDA_ACTIVATE=
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" set CONDA_ACTIVATE=C:\ProgramData\anaconda3\Scripts\activate.bat
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat

if "%CONDA_ACTIVATE%"=="" echo [ERRO] Nao foi possivel localizar o activate.bat do Anaconda.
if "%CONDA_ACTIVATE%"=="" pause
if "%CONDA_ACTIVATE%"=="" exit /b 1

echo [AETHER] Etapa 0/4: Sanitizando run anterior (checkpoints/meta/loss/triton/data)...
cd /d "%REPO_ROOT%"
"%PYTHON_EXE%" app/scripts/operations/sanitize_fresh_run.py
if errorlevel 1 goto :sanitize_fail

echo [AETHER] Etapa 0b/4: Regenerando bootstrap do loss-classifier...
cd /d "%REPO_ROOT%\app"
"%PYTHON_EXE%" -m scripts.operations.train_loss_classifier
if errorlevel 1 goto :loss_bootstrap_fail
cd /d "%REPO_ROOT%"

echo [AETHER] Etapa 1/4: Executando treino Deep Learning (TCN)...
call "%~dp0_run_train.bat" "%CONDA_ACTIVATE%"
if errorlevel 1 goto :dl_fail

echo [AETHER] Etapa 1b/4: Validando deploy/ACC do checkpoint DL...
cd /d "%REPO_ROOT%"
"%PYTHON_EXE%" app/scripts/operations/check_dl_deploy_gate.py --symbols R_10
if errorlevel 1 goto :dl_gate_fail

echo [AETHER] Etapa 2/4: Verificando infraestrutura Docker (TimescaleDB)...
cd /d "%REPO_ROOT%"
"%PYTHON_EXE%" app/scripts/operations/ensure_timescale.py --check-only
if errorlevel 1 echo [AVISO] TimescaleDB nao disponivel. Meta-classificador usara API Deriv (fallback).

echo [AETHER] Etapa 3/4: Treino DL concluido com sucesso! Iniciando Meta-Classificador (LightGBM)...
call "%~dp0_run_meta_train.bat" "%CONDA_ACTIVATE%"
if errorlevel 1 goto :meta_fail

echo [SUCESSO] Pipeline de treinamento AETHER concluido com sucesso!
timeout /t 5 /nobreak > nul
exit /b 0

:sanitize_fail
echo [ERRO] Sanitizacao da run anterior falhou. Treino abortado.
pause
exit /b 1

:loss_bootstrap_fail
echo [ERRO] Bootstrap do loss-classifier falhou apos sanitizacao. Treino abortado.
pause
exit /b 1

:dl_fail
echo [ERRO] Treino DL falhou. Treino do Meta-Classificador abortado para evitar race condition.
pause
exit /b 1

:dl_gate_fail
echo [ERRO] Checkpoint DL sem deploy_ok/ACC>=0.53. Meta-Classificador abortado.
pause
exit /b 1

:meta_fail
echo [ERRO] Treino do Meta-Classificador falhou.
pause
exit /b 1
