@echo off
setlocal EnableExtensions
pushd "%~dp0..\.."
set "APP_ROOT=%CD%"
popd
set "ENV_NAME=deriv-api"
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
cd /d "%APP_ROOT%"
echo [AETHER] Motor - conda=%ENV_NAME% python=%PYTHON_EXE% dir=%CD%
"%PYTHON_EXE%" run.py
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [ERRO] Motor encerrou com codigo %RC%.
    pause
)
endlocal
exit /b %RC%
