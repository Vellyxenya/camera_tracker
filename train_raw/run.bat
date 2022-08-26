echo Run Tesseract for Training.. 
C:\Users\Gueddach\AppData\Local\Tesseract-OCR\tesseract.exe num.font.exp0.tif num.font.exp0 nobatch box.train 
 
echo Compute the Character Set.. 
C:\Users\Gueddach\AppData\Local\Tesseract-OCR\unicharset_extractor.exe num.font.exp0.box 
C:\Users\Gueddach\AppData\Local\Tesseract-OCR\shapeclustering -F font_properties -U unicharset -O num.unicharset num.font.exp0.tr
C:\Users\Gueddach\AppData\Local\Tesseract-OCR\mftraining -F font_properties -U unicharset -O num.unicharset num.font.exp0.tr
echo Clustering.. 
C:\Users\Gueddach\AppData\Local\Tesseract-OCR\cntraining.exe num.font.exp0.tr
echo Rename Files.. 
rename normproto num.normproto 
rename inttemp num.inttemp 
rename pffmtable num.pffmtable 
rename shapetable num.shapetable
echo Create Tessdata.. 
C:\Users\Gueddach\AppData\Local\Tesseract-OCR\combine_tessdata.exe num.
echo. & pause