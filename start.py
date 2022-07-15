from paddleocr import PaddleOCR
import numpy as np

#ocr = PaddleOCR(use_gpu=False, lang="ch", type="ocr", det_db_box_thresh=0.02, det_db_thresh=0.02, det_db_unclip_ratio=1.4, max_batch_size=100, use_mp=True, drop_score = 0.2)
#PaddleOCR(use_angle_cls=True, lang="ch", det_limit_type='min', det_limit_side_len=64)

ocr1 = PaddleOCR(use_gpu=False, use_angle_cls=True, lang="ch", type="ocr", det_db_box_thresh=0.02, det_db_thresh=0.02, det_db_unclip_ratio=0.1, max_batch_size=100, use_mp=True, det_limit_type='min', det_limit_side_len=64)




#ocr = PaddleOCR(use_gpu=False, lang="ch", type="ocr", det_db_box_thresh=0.02, det_db_thresh=0.02, det_db_unclip_ratio=1.4, max_batch_size=100, use_mp=True, drop_score = 0.2)

img_path = "test_noise.jpeg"
#img_path ="https://img.alicdn.com/bao/uploaded/i3/2597705728/O1CN01jSbQBn1sBTPzT80Dc_!!0-item_pic.jpg_430x430q90.jpg"
#img_path = "/home/ec2-user/SageMaker/OCR/paddleocr/imgs/tmallsample1.png"


result = ocr1.ocr(img_path, cls=False)

#txts = [line[1][0] for line in result]


# texts = []
# if result is not None:
#     for i in range(len(result)):
#         _boxes = result[i]
    #result = sorted(result, key=lambda x: x[0][0][1])
    #print('result',result)

#     num_boxes = np.array(result).shape[0]
#     _boxes = result
#     for i in range(1, num_boxes):
#         if abs(_boxes[i][0][0][1] - _boxes[i - 1][0][0][1]) <= 40 and \
#                     (_boxes[i-1][0][0][0] > _boxes[i][0][0][0]):
#                 tmp = _boxes[i-1]
#                 _boxes[i-1] = _boxes[i]
#                 _boxes[i] = tmp

#         for i in range(1, num_boxes):
#             if abs(_boxes[i][0][0][1] - _boxes[i - 1][0][0][1]) <= 40 and \
#                     (_boxes[i-1][0][0][0] > _boxes[i][0][0][0]):
#                 tmp = _boxes[i-1]
#                 _boxes[i-1] = _boxes[i]
#                 _boxes[i] = tmp

#         for i in range(1, num_boxes):
#             if abs(_boxes[i][0][0][1] - _boxes[i - 1][0][0][1]) <= 40 and \
#                     (_boxes[i-1][0][0][0] > _boxes[i][0][0][0]):
#                 tmp = _boxes[i-1]
#                 _boxes[i-1] = _boxes[i]
#                 _boxes[i] = tmp
#     swapped = True
#     num_of_iterations = 0
#     while swapped:
#         swapped = False
      
#         for i in range(num_boxes - num_of_iterations - 1):
#             if abs(_boxes[i+1][0][0][1] - _boxes[i][0][0][1]) <= 50 and (_boxes[i][0][0][0] > _boxes[i+1][0][0][0]):
#                 tmp = _boxes[i]
#                 _boxes[i] = _boxes[i+1]
#                 _boxes[i] = tmp
#                 #_boxes[i], _boxes[i+1] = _boxes[i+1], _boxes[i]
#             #print('-------',_boxes[i][0][0][1], _boxes[i])
#             swapped = True
#         num_of_iterations += 1
    #print(_boxes)
    
#     texts.append(_boxes)
    
#print(texts)

print(result)


with open('/home/ec2-user/SageMaker/OCR/paddleocr/outputs/MedReport/tmallsample1_4.txt', 'w') as f:
    for line in result:
        #print(line)
        f.write(str(line[1]))
        f.write('\n')

        
   
    
# export LD_LIBRARY_PATH=/home/ec2-user/anaconda3/envs/python3/lib
# with open('/home/ec2-user/SageMaker/OCR/paddleocr/outputs/MedReport/tmallsample1.txt', 'w') as f:
#     for line in txts:
#         f.write(line)
#         f.write('\n')


#print(txts)