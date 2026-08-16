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

echo [AETHER] launch-train: sanitize -^> sweep H15..H60 -^> gate -^> meta -^> rebuild
echo [AETHER] 0/5 sanitize run anterior...
cd /d "%REPO_ROOT%"
"%PYTHON_EXE%" app/scripts/operations/sanitize_fresh_run.py
if errorlevel 1 goto :sanitize_fail

echo [AETHER] 0b/5 loss-classifier bootstrap...
cd /d "%REPO_ROOT%\app"
"%PYTHON_EXE%" -m scripts.operations.train_loss_classifier
if errorlevel 1 goto :loss_bootstrap_fail
cd /d "%REPO_ROOT%"

echo [AETHER] 1/5 sweep horizonte + promote (ops duration M5 fixo)...
cd /d "%REPO_ROOT%"
"%PYTHON_EXE%" app/scripts/operations/run_launch_train_tf_pipeline.py %*
if errorlevel 1 goto :horizon_fail

echo [AETHER] 1b/5 gate deploy/settle do checkpoint...
cd /d "%REPO_ROOT%"
"%PYTHON_EXE%" app/scripts/operations/check_dl_deploy_gate.py
if errorlevel 1 goto :dl_gate_fail

echo [AETHER] 2/5 Timescale check-only...
cd /d "%REPO_ROOT%"
"%PYTHON_EXE%" app/scripts/operations/ensure_timescale.py --check-only
if errorlevel 1 echo [AVISO] Timescale indisponivel; meta usara API Deriv.

echo [AETHER] 3/5 meta LightGBM...
call "%~dp0_run_meta_train.bat" "%CONDA_ACTIVATE%"
if errorlevel 1 goto :meta_fail

echo [SUCESSO] launch-train OK (TCN + meta).
echo [AETHER] Proximo: make docker-rebuild + sync MinIO, depois DEMO.
timeout /t 5 /nobreak > nul
exit /b 0

:sanitize_fail
echo [ERRO] Sanitize falhou; treino abortado.
pause
exit /b 1

:loss_bootstrap_fail
echo [ERRO] Bootstrap loss-classifier falhou; treino abortado.
pause
exit /b 1

:horizon_fail
echo [ERRO] Sweep/promote falhou. Criterio: settle_wr^>=be+0.03, n^>=16, history^>=800.
pause
exit /b 1

:dl_gate_fail
echo [ERRO] Checkpoint sem deploy_ok/settle elegivel; meta abortado.
pause
exit /b 1

:meta_fail
echo [ERRO] Meta-classificador falhou.
pause
exit /b 1
