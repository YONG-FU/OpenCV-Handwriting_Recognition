@echo off

set "uirobotpath=C:\Program Files (x86)\UiPath Platform\UiRobot.exe"
set "uiprocesspath=C:\Users\yong-fu\PycharmProjects\Text-Recognition\TextRecognition\Main_CN.xaml"


cd C:\Users\yong-fu\PycharmProjects\Text-Recognition\Demo
python linedetect_sliceall.py
python VerticalSlice.py


"%uirobotpath%" /file "%uiprocesspath%"

timeout /t 30 