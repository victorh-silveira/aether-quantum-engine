@echo off
setlocal EnableExtensions
pushd "%~dp0..\.."
set "APP_ROOT=%CD%"
popd
if not "%~1"=="" (
    call "%~1" deriv-api
    if errorlevel 1 (
        echo [ERRO] Falha ao ativar ambiente Conda deriv-api.
        pause
        exit /b 1
    )
)
cd /d "%APP_ROOT%"
echo [AETHER] Motor - diretorio=%CD%
python run.py
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [ERRO] Motor encerrou com codigo %RC%.
    pause
)
endlocal
exit /b %RC%
