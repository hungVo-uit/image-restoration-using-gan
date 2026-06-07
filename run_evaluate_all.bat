@echo off

cd /d REAL-ESRGAN
py predict.py

cd /d ..\DeblurGANv2
py predict.py

@REM cd /d ..\data
@REM py evaluate.py

cd /d ..
pause