import os,cv2
import numpy as np

src_path = "/home/zrh/datasets/datasets_org/vaihingen/"
save_path = "/home/zrh/datasets/datasets_org/vaihingen/2/"

vaihingen_splits = {
        'train': [
            'area1', 'area3', 'area5', 'area7', 
            'area11', 'area13','area15', 'area17', 
            'area21', 'area23', 'area26', 'area28',
            'area30', 'area32', 'area34', 'area37'],
        'test': [
            'area2', 'area4', 'area6', 'area8', 'area10', 
            'area12','area14', 'area16', 'area20', 
            'area22', 'area24', 'area27', 'area29',
            'area31', 'area33', 'area35', 'area38']}

prefix = "top_mosaic_09cm_"
mode = 'train'

for name_id in vaihingen_splits[mode]:
    pure_name = prefix+name_id + ".tif"
    #print(pure_name)
    img_name = src_path+"image/"+pure_name
    #print(img_name)
    label_name = src_path+"label/"+pure_name

    img = cv2.imread(img_name,cv2.IMREAD_UNCHANGED)
    print(img.shape)
    label = cv2.imread(label_name,cv2.IMREAD_UNCHANGED)
    print(label.shape)

    cv2.imwrite(save_path+mode+"/images/"+pure_name,img)
    cv2.imwrite(save_path+mode+"/masks/"+pure_name,label)

