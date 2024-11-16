Cell Counter Telegram Bot

https://t.me/cell_counter_dstu_bot 

This Telegram bot can count cells -- in the reply message it sends the number of detected objects and the image with segmentation mask on it 

We generated a dataset that is available https://app.roboflow.com/cultures/cell-segmentation-ci5n3/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true. 
It contains three cell cultures:
Culture 1 -- Adiposytes
Culture 2 -- Fibroblasts
Culture 3 -- Mesenchymal stem cells
All our models were finetuned only on this dataset


Bot has 4 models:
three YOLOv8-seg models that were finetuned only on one cell culture and on that was finetuned on all three cultures