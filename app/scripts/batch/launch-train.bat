@echo off
TITLE Aether Engine - Treino Deep Learning + Meta-Classificador
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

set CONDA_ACTIVATE=
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" set CONDA_ACTIVATE=C:\ProgramData\anaconda3\Scripts\activate.bat
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat

if "%CONDA_ACTIVATE%"=="" echo [ERRO] Nao foi possivel localizar o activate.bat do Anaconda.
if "%CONDA_ACTIVATE%"=="" pause
if "%CONDA_ACTIVATE%"=="" exit /b 1

echo [AETHER] launch-train: sanitize -^> treino DL 5m -^> gate -^> meta -^> rebuild
echo [AETHER] 0/5 sanitize run anterior...
cd /d "%REPO_ROOT%"
"%PYTHON_EXE%" -u app/scripts/operations/sanitize_fresh_run.py
if errorlevel 1 goto :sanitize_fail

echo [AETHER] 0b/5 loss-classifier bootstrap...
cd /d "%REPO_ROOT%\app"
"%PYTHON_EXE%" -u -m scripts.operations.train_loss_classifier
if errorlevel 1 goto :loss_bootstrap_fail
cd /d "%REPO_ROOT%"

echo [AETHER] 1/5 treino TCN direto (contrato M5 fixo)...
cd /d "%REPO_ROOT%"
"%PYTHON_EXE%" -u app/scripts/operations/run_launch_train_tf_pipeline.py %*
if errorlevel 1 goto :horizon_fail

echo [AETHER] 1b/5 gate preliminar TCN (feature extractor)...
cd /d "%REPO_ROOT%"
"%PYTHON_EXE%" -u app/scripts/operations/check_dl_deploy_gate.py --allow-unqualified
if errorlevel 1 (
    echo [AVISO] Checkpoint TCN treinado como feature extractor; avaliacao conjunta dependente do meta.
)

echo [AETHER] 2/5 Timescale seed meta-ready (Deriv se smoke/curto)...
cd /d "%REPO_ROOT%"
"%PYTHON_EXE%" -u app/scripts/operations/ensure_timescale.py
if errorlevel 1 echo [AVISO] Timescale seed falhou; meta usara API Deriv.

echo [AETHER] 3/5 meta LightGBM...
call "%~dp0_run_meta_train.bat" "%CONDA_ACTIVATE%"
if errorlevel 1 goto :meta_fail
set "META_READY=1"
if not exist "%REPO_ROOT%\infra\docker\meta-models\meta_lgbm.pkl" (
    if exist "%REPO_ROOT%\data\dl\meta_candidate.joblib" (
        copy /y "%REPO_ROOT%\data\dl\meta_candidate.joblib" "%REPO_ROOT%\infra\docker\meta-models\meta_lgbm.pkl" > nul
        echo [AVISO] Modelo meta candidato promovido como baseline local em meta-models/meta_lgbm.pkl.
    ) else (
        set "META_READY=0"
    )
)

echo [AETHER] 4/5 gate deploy conjunto Two-Stage Stacking (TCN + Meta)...
cd /d "%REPO_ROOT%"
set "DEPLOY_READY=1"
"%PYTHON_EXE%" -u app/scripts/operations/check_dl_deploy_gate.py --with-meta
if errorlevel 1 (
    set "DEPLOY_READY=0"
    echo [AVISO] Gate conjunto Two-Stage Stacking reprovado; operando em modo protegido.
)

if "%DEPLOY_READY%"=="0" goto :summary_unqualified
if "%META_READY%"=="0" goto :summary_meta_candidate
echo [SUCESSO] launch-train OK: TCN e meta com deploy aprovado.
echo [AETHER] Proximo: make docker-rebuild + sync MinIO, depois DEMO.
goto :launch_train_done

:summary_unqualified
echo [SUCESSO] Treino TCN + meta concluido; TCN nao qualificado, checkpoint local opera com teto de 0,1%% em DEMO e REAL.
echo [AETHER] O gate OOS ainda nao demonstrou vantagem preditiva.
goto :launch_train_done

:summary_meta_candidate
echo [AVISO] TCN aprovado, mas meta apenas candidato; nao faca rebuild para trading.
goto :launch_train_done

:launch_train_done
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
echo [ERRO] Treino TCN falhou antes de exportar checkpoint; veja logs.
pause
exit /b 1

:meta_fail
echo [ERRO] Meta-classificador falhou.
pause
exit /b 1
