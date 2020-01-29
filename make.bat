@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%1" == "" goto install

goto %1

:clean-docs
echo.Cleaning docs...
call docs\make clean
goto end


:docs
call docs\make html
goto end

:doc
call docs\make clean
call docs\make html
start .\docs\build\html\index.html
goto end

:format
isort --recursive pyraman
black -t py36 pyraman
docformatter --in-place --recursive pyraman
goto end

:lint
flake8 pyraman
pylint pyraman
goto end


:install
pip install -r requirements.txt
pip install -e .
goto end


:test
mkdir tests\reports
python -m pytest

:end
popd
