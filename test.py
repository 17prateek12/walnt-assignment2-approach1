from mmdet.apis import init_detector, inference_detector
import mmcv
import fitz
from openpyxl import Workbook
import pandas as pd


config_file = 'D:\python project\walnut ai assignment 2.1\CascadeTabNet\Config\cascade_mask_rcnn_hrnetv2p_w32_20e.py'
checkpoint_file = 'epoch_36.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')


pdf_path="D:\python project\walnut ai assignement 2.1\keppel-corporation-limited-annual-report-2018.pdf"
page_no=68
excel_path = "D:\python project\walnut ai assignement 2.1\document.xlsx"

with fitz.open(pdf_path) as pdf:
    page = pdf.load_page(page_no)
    extracted_img = page.get_pixmap()
    
img_path = "D:\python project\walnut ai assignment 2.1\image.png"
img=extracted_img.save(img_path)

result = inference_detector(model, img)
print(result)
#show_result_pyplot(img, result,('Bordered', 'cell', 'Borderless'), score_thr=0.85,show=False)

