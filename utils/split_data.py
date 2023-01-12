import os
import random

base_path = '/home/mrjiang/Server_mrjiang/Dataset/OfficeHomeDataset'
OfficeHome = ['art', 'clipart', 'product', 'real_world']
# OfficeHome = ['art',]
class_list = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam']
class_dict = dict(enumerate(class_list))
class_dict = dict((v,k) for k,v in class_dict.items())

data_dict = {}
for d in OfficeHome:
    data_dict[d] = []
    folder = os.path.join(base_path, d)
    classes = os.listdir(folder)
    for c in classes:
        c_path = os.path.join(base_path, d , c)
        imgs = os.listdir(c_path)
        for img in imgs:
            img_path = os.path.join(c_path, img)
            sample_pair = (img_path, class_dict[c])
            data_dict[d].append(sample_pair)

for k,v in data_dict.items():
    random.shuffle(v)
    split_len = int(len(v)*0.8) 
    train = v[:split_len]
    test = v[split_len:]
    with open(f'{k}_train.txt', 'w') as f:
        for pair in train:
            f.write(f'{pair[0]} {int(pair[1])}\n')

    with open(f'{k}_test.txt', 'w') as f:
        for pair in test:
            f.write(f'{pair[0]} {int(pair[1])}\n')

