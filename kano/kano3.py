import pandas as pd
from paddleocr import PaddleOCR
import numpy as np
import time 
from openpyxl import load_workbook
import urllib.request
import urllib.parse
import os
from urllib.error import HTTPError


start_time = time.time()


wb = load_workbook('kano_pocr_v3.xlsx')
wb.save(filename = 'temp.xlsx')
df = pd.read_excel('temp.xlsx', engine='openpyxl')

#df = pd.read_excel('kano_pocr_v1.xlsx', engine = 'openpyxl')


df_empty = df[df['pocr'].isnull()]
empty_list = list(df_empty.index)

#print(df_empty.head())



ocr = PaddleOCR(use_gpu=False, lang="ch", type="structure", det_db_box_thresh=0.02, det_db_thresh=0.02, det_db_unclip_ratio=3.0, max_batch_size=100, use_mp=True)


a = 0 
texts = []
for i in list(df_empty['image_url']):
    url = i
    name = empty_list[a]
    fullname = "/home/ec2-user/SageMaker/OCR/paddleocr/kano/"+str(name)+".jpg"
    try: 
        urllib.request.urlretrieve(url,fullname) 
    except ConnectionResetError:
        time.sleep(5)
    except urllib.error.HTTPError as err:
        print(url)
        url = urllib.parse.quote(url,':/=&?')
        pass
        #urllib.request.urlretrieve(url,fullname)
  
    img_path = fullname
    
    index = empty_list[a]
    output = open('/home/ec2-user/SageMaker/OCR/paddleocr/outputs/MedReport/kano_raw_hair_a7.txt', 'a+')
    output.writelines(str(index) + ",       ")
    try:
        result = ocr.ocr(img_path, cls=False)
        if result is not None:
            result = sorted(result, key=lambda x: x[0][0][1])

            num_boxes = np.array(result).shape[0]
            _boxes = result

            for i in range(1, num_boxes):
                if abs(_boxes[i][0][0][1] - _boxes[i - 1][0][0][1]) <= 16 and \
                        (_boxes[i-1][0][0][0] > _boxes[i][0][0][0]):
                    tmp = _boxes[i-1]
                    _boxes[i-1] = _boxes[i]
                    _boxes[i] = tmp

            for i in range(1, num_boxes):
                if abs(_boxes[i][0][0][1] - _boxes[i - 1][0][0][1]) <= 15 and \
                        (_boxes[i-1][0][0][0] > _boxes[i][0][0][0]):
                    tmp = _boxes[i-1]
                    _boxes[i-1] = _boxes[i]
                    _boxes[i] = tmp

            for i in range(1, num_boxes):
                if abs(_boxes[i][0][0][1] - _boxes[i - 1][0][0][1]) <= 15 and \
                        (_boxes[i-1][0][0][0] > _boxes[i][0][0][0]):
                    tmp = _boxes[i-1]
                    _boxes[i-1] = _boxes[i]
                    _boxes[i] = tmp

            result = _boxes

   
            for line in result:
                output = open('/home/ec2-user/SageMaker/OCR/paddleocr/outputs/MedReport/kano_raw_hair_a7.txt', 'a+')
                if line[1][0] is not None:
                    output.writelines(str(line[1][0]) + ",")
#                 else:
#                     text = ""
#                     output.writelines(text)

        else:
            text = ""
            output.writelines(text)
    except: 
        pass
    output.writelines('\n')
    try:
        os.remove(fullname)
    except:
        pass
    print(empty_list[a])
    #print(i)
    #print(result)
    print("--- %s seconds ---" % (time.time() - start_time))
    a = a+1
    
    
  
                
    
    


   