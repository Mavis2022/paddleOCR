import pandas as pd
from paddleocr import PaddleOCR
import numpy as np
import time 

start_time = time.time()

df = pd.read_excel('KANO_hair.xlsx', engine = 'openpyxl')

ocr = PaddleOCR(use_gpu=False, lang="ch", type="structure", det_db_box_thresh=0.05, det_db_thresh=0.05, det_db_unclip_ratio=3.0, max_batch_size=10, use_mp=True)

ocr1 = PaddleOCR(use_gpu=False, lang="ch", type="structure", det_db_box_thresh=0.05, det_db_thresh=0.05, det_db_unclip_ratio=1.0, max_batch_size=10, use_mp=True)

ocr2 = PaddleOCR(use_gpu=False, lang="ch", type="structure", det_db_box_thresh=0.05, det_db_thresh=0.05, det_db_unclip_ratio=0.5, max_batch_size=10, use_mp=True)


a = 0 
texts = []
for i in list(df['image_url']):
    img_path = i
    index = a
    output = open('/home/ec2-user/SageMaker/OCR/paddleocr/outputs/MedReport/kano_raw_hair_3.txt', 'a+')
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
                output = open('/home/ec2-user/SageMaker/OCR/paddleocr/outputs/MedReport/kano_raw_hair_3.txt', 'a+')
                if line[1][0] is not None:
                    output.writelines(str(line[1][0]) + ",")
#                 else:
#                     text = ""
#                     output.writelines(text)

        else:
            result = ocr1.ocr(img_path, cls=False)
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
                    output = open('/home/ec2-user/SageMaker/OCR/paddleocr/outputs/MedReport/kano_raw_hair_3.txt', 'a+')
                    if line[1][0] is not None:
                        output.writelines(str(line[1][0]) + ",")
            else:
                result = ocr2.ocr(img_path, cls=False)
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
                        output = open('/home/ec2-user/SageMaker/OCR/paddleocr/outputs/MedReport/kano_raw_hair_3.txt', 'a+')
                        if line[1][0] is not None:
                            output.writelines(str(line[1][0]) + ",")
             
    except: 
        pass
    output.writelines('\n')
    print(a)
    #print(i)
    #print(result)
    print("--- %s seconds ---" % (time.time() - start_time))
    a = a+1
    
    
  
                
    
    