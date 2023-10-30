# Copyright from Sharebee.cn Inc All rights Reserved.
# Author: Samuel
# Date: April 2, 2023
# Reference: https://github.com/Samuel-wei/Alexey-darknet/blob/master/gen_files.py and GPT

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
import matplotlib
import shutil 

#classes = ["Scratch", "PitCrushed", "Crushed", "RustCurrosion", "Wound"]
classes = ["defect"]
# object = "battery"

def clear_hidden_files(path):
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
     # The math.fabs() function to get the number absolute value
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return(x, y, w, h)

def sep_test(file):
    name_extention = file.split('.')[-1]
    if probo < 0: #trian=101 test=0 trian/test=75
        if name_extention == 'jpg':
            shutil.copy(file,test_img_dst_path)
        if name_extention == 'xml':
            shutil.copy(file,test_xml_dst_path)      
    else:
        if name_extention == 'jpg':
            shutil.copy(file,test_img_dst_path)
        if name_extention == 'xml':
            shutil.copy(file,test_xml_dst_path)


def convert_annotation(image_id):
    if probo < 0:  #trian=101 test=0 trian/test=75
        in_file = open('/home/workspace/BatteryDataSet/Data_xml/TestData/NG_14_20220907_xml/%s.xml'%image_id)
        out_file = open('/home/workspace/BatteryDataSet/DataCOCO/labels/test-dev2017/%s.txt'%image_id, 'w')

        
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        in_file.close()
        out_file.close()

    else:				
        in_file = open('/home/workspace/BatteryDataSet/Data_xml/TestData/NG_14_20220907_xml/%s.xml'%image_id)
        out_file = open('/home/workspace/BatteryDataSet/DataCOCO/labels/test-dev2017/%s.txt'%image_id, 'w')


        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        in_file.close()
        out_file.close()
    

# Check path and input parameters
#wd = os.getcwd()
wd = '/home/workspace/'
print(wd)

# work_space_dir = os.path.join(wd)
work_space_dir = os.path.join(wd, 'BatteryDataSet/Data_xml/TestData/')
if not os.path.isdir(work_space_dir):
    os.makedirs(work_space_dir)
image_dir = os.path.join(work_space_dir, "NG_14_20220907/")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
clear_hidden_files(image_dir)
annotation_dir = os.path.join(work_space_dir, "NG_14_20220907_xml/")
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)
clear_hidden_files(annotation_dir)

# train_file = open(os.path.join(work_space_dir, "train2017.txt"), 'w')
test_file = open(os.path.join(wd, "BatteryDataSet/DataCOCO/test-dev2017.txt"), 'w')
# train_file.close()
test_file.close()

# labels_train = '/home/workspace/BatteryDetect/DataCOCO/ValidataionData/labels/train2017/'
# if not os.path.isdir(labels_train):
#         os.makedirs(labels_train)
labels_test = '/home/workspace/BatteryDataSet/DataCOCO/labels/test-dev2017/'
if not os.path.isdir(labels_test):
        os.makedirs(labels_test)

# train_img_dst_path = "/home/workspace/BatteryDetect/DataCOCO/ValidataionData/images/train2017/"
# if not os.path.isdir(train_img_dst_path):
#     os.makedirs(train_img_dst_path)
# train_xml_dst_path = "/home/workspace/BatteryDetect/DataCOCO/ValidataionData/xmls/train2017/"
# if not os.path.isdir(train_xml_dst_path):
#     os.makedirs(train_xml_dst_path)

test_img_dst_path = '/home/workspace/BatteryDataSet/DataCOCO/images/test-dev2017/'
if not os.path.isdir(test_img_dst_path):
    os.makedirs(test_img_dst_path)
test_xml_dst_path = '/home/workspace/BatteryDataSet/DataCOCO/xmls/test-dev2017/'
if not os.path.isdir(test_xml_dst_path):
    os.makedirs(test_xml_dst_path)


# train_file = open(os.path.join(work_space_dir, "train2017.txt"), 'a')
test_file = open(os.path.join(wd, "BatteryDataSet/DataCOCO/test-dev2017.txt"), 'a')
	
list = os.listdir(image_dir) # list image files
probo = random.randint(1, 100)
print("Probobily: %d" % probo)
for i in range(0, len(list)):
    path = os.path.join(image_dir, list[i])
    if os.path.isfile(path):
        image_path = image_dir + list[i]
        # train_path = train_img_dst_path + list[i]
        test_path = test_img_dst_path + list[i]
        voc_path = list[i]
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
        (voc_nameWithoutExtention, voc_extention) = os.path.splitext(os.path.basename(voc_path))
        annotation_name = nameWithoutExtention + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_name)
    probo = random.randint(1, 100)
    print("probobility: %d" % probo)
    if (probo < 0): #trian=101 test=0 trian/test=75
        if os.path.exists(annotation_path):
            # train_file.write(image_path + '\n')
            sep_test(annotation_path)
            sep_test(image_path)
            # train_file.write(train_path + '\n')
            convert_annotation(nameWithoutExtention)
    else:
        if os.path.exists(annotation_path):
            # val_file.write(image_path + '\n')  
            sep_test(annotation_path)
            sep_test(image_path)
            test_file.write(test_path + '\n')
            convert_annotation(nameWithoutExtention)
# train_file.close()
test_file.close()
