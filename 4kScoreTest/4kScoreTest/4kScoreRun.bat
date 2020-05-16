del /s /q C:\Users\SJTU-VQA\Desktop\4kScoreTest\4kScoreTest\information.txt
rd /s /q C:\Users\SJTU-VQA\Desktop\4kScoreTest\4kScoreTest\data\imgSlice
md C:\Users\SJTU-VQA\Desktop\4kScoreTest\4kScoreTest\data\imgSlice
ffmpeg -i C:\Users\SJTU-VQA\Desktop\4kScoreTest\4kScoreTest\data\demo.mxf  -r 1 -ss 00:00:00 -t 00:00:07 -q:v 2 C:\Users\SJTU-VQA\Desktop\4kScoreTest\4kScoreTest\data\imgSlice\pic%%03d.bmp
start C:\Users\SJTU-VQA\Desktop\4kScoreTest\4kScoreTest\4kScoreTest.exe
pause