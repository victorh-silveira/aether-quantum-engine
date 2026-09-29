@echo off
TITLE Aether Engine - Treino Rise-Fall TCN + Meta
set PYTHONASYNCIODEBUG=
set PYTHONDEVMODE=
set PYTHONUNBUFFERED=1

pushd "%~dp0..\..\.."
SET REPO_ROOT=%CD%
popd
SET ENV_NAME=deriv-api
set PYTHON_EXE=%USERPROFILE%\anaconda3\envs\%ENV_NAME%\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=%USERPROFILE%\miniconda3\envs\%ENV_NAME%\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

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

"%PYTHON_EXE%" -u app/scripts/operations/run_launch_train_tf_pipeline.py %*
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" -u app/scripts/operations/check_dl_deploy_gate.py --allow-unqualified
if errorlevel 1 echo [AVISO] TCN sem qualificacao de deploy; checkpoint local permanece sujeito ao teto de risco.

"%PYTHON_EXE%" -u app/scripts/operations/ensure_timescale.py
if errorlevel 1 echo [AVISO] Timescale seed indisponivel; meta tentara API Deriv.

"%PYTHON_EXE%" -u app/scripts/operations/train_meta_classifier.py --trials 60 --bars 5000 --source auto --candidate-on-low-quality
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" -u app/scripts/operations/check_dl_deploy_gate.py --with-meta
if errorlevel 1 (
    if exist "%REPO_ROOT%\infra\docker\meta-models\meta_lgbm.pkl" (
        echo [AVISO] Meta exportado em meta-models, mas gate conjunto TCN + meta nao qualificado.
    ) else (
        echo [AVISO] Meta nao exportado; somente candidato diagnostico. Gate conjunto TCN + meta nao qualificado.
    )
) else (
    echo [AETHER] TCN + meta Rise/Fall qualificados.
)
exit /b 0
