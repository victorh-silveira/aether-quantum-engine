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
echo [AETHER] Monitor - diretorio=%CD%
python -m scripts.monitor.live_monitor
if errorlevel 1 (
    echo.
    echo [ERRO] Monitor encerrou com erro.
    pause
)
endlocal
