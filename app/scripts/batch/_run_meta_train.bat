@echo off
setlocal EnableExtensions
set "PYTHONASYNCIODEBUG="
set "PYTHONDEVMODE="
pushd "%~dp0..\..\.."
set "REPO_ROOT=%CD%"
popd
set "ENV_NAME=deriv-api"
set "META_TRIALS=60"
set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\%ENV_NAME%\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\%ENV_NAME%\python.exe"
if not exist "%PYTHON_EXE%" (
    if not "%~1"=="" (
        call "%~1" %ENV_NAME%
        if errorlevel 1 (
            echo [ERRO] Falha ao ativar ambiente Conda %ENV_NAME%.
            pause
            exit /b 1
        )
        set "PYTHON_EXE=python"
    ) else (
        echo [ERRO] Python do Conda %ENV_NAME% nao encontrado.
        pause
        exit /b 1
    )
)
cd /d "%REPO_ROOT%"
echo [AETHER] Meta-classificador trials=%META_TRIALS% source=auto bars=5000
"%PYTHON_EXE%" app/scripts/operations/train_meta_classifier.py --trials %META_TRIALS% --bars 5000 --source auto
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [ERRO] Treino meta-classificador encerrou com codigo %RC%.
    pause
)
endlocal
exit /b %RC%
