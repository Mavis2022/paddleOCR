import os
import urllib.request
import urllib.parse
import urllib
import time
import pandas as pd

df = pd.read_excel('kano.xlsx')

row_num, col_num = df.shape[0], df.shape[1]

for id in range(row_num):
    if (str(id) + ".jpg") not in os.listdir():
        fullname = "~/OCR/paddleocr/kano/kano_imgs/" + str(id) + ".jpg"
        # store image url
        url = df['image_url'][id]
        try:
            urllib.request.urlretrieve(url, fullname)
        except ConnectionResetError:
            time.sleep(5)
        except urllib.error.HTTPError as err:
            url = urllib.parse.quote(url, ':/=&?')
            # urllib.request.urlretrieve(url, fullname)
            continue
        except urllib.error.URLError as err:
            continue
    else:
        continue
        
        
 
