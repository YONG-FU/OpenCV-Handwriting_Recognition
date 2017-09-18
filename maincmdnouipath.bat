@echo off

set "uirobotpath=C:\Program Files (x86)\UiPath Studio\UiRobot.exe"
set "uiprocesspath=C:\Users\yyi012\Desktop\PythonDemo\TextRecognition\Main.xaml"


cd C:\Users\yyi012\Desktop\PythonDemo\Demo
python linedetect_sliceall.py
python VerticalSlice.py


@rem "%uirobotpath%" /file "%uiprocesspath%"

timeout /t 30 