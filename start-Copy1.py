from paddleocr import PaddleOCR
import numpy as np



#ocr = PaddleOCR(use_gpu=False, lang="ch", type="ocr", det_db_box_thresh=0.02, det_db_thresh=0.02, det_db_unclip_ratio=1.4, max_batch_size=100, use_mp=True, drop_score = 0.2)


ocr = PaddleOCR(use_gpu=False, lang="ch", type="structure", det_db_box_thresh=0.05, det_db_thresh=0.05, det_db_unclip_ratio=0.1, max_batch_size=10, use_mp=True)

img_path = "https://img.alicdn.com/bao/uploaded/i2/6000000001567/O1CN01eXpKqu1NRjNva0iK5_!!6000000001567-0-picassoopen.jpg"

#img_path = "/home/ec2-user/SageMaker/OCR/paddleocr/imgs/tmallsample1.png"


result = ocr.ocr(img_path, cls=False)




with open('/home/ec2-user/SageMaker/OCR/paddleocr/outputs/MedReport/tmallsample1_4.txt', 'w') as f:
    for line in result:
        #print(line)
        f.write(str(line[1]))
        f.write('\n')

        
   
    
