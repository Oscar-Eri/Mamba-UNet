import  os
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from PIL import Image

class DataCollector(object):
    def __init__(self, root='data',output_folder="split", **kwargs):

        self.dataset_dir=root
        self.output_folder=output_folder
        self.train_datasets,self.test_datasets,self.len_train,self.len_test=self.date_set()

        #打印信息
        print("  ------------------------------")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # nums |")
        print("  ------------------------------")
        print("  train    | {:5d}  |".format(self.len_train))
        print("  test     | {:5d}  |".format(self.len_test))
        print("  ------------------------------")
        print("  total    | {:5d}  |".format(self.len_train+self.len_test))
        print("  ------------------------------")


    def date_set(self):
        list_dirs=os.listdir(self.dataset_dir)
        print("=>" + self.dataset_dir + " loaded")

        img_num_test = int(len(list_dirs) * 0.0) #测试集百分比#可设置，原始值为0
        random.seed(10) #  随机种子
        list_test = random.sample(list_dirs,img_num_test)
        list_train=list(set(list_dirs)-set(list_test))
        list_test = [*map(lambda x: os.path.join(self.dataset_dir, x), list_test)]
        list_train = [*map(lambda x: os.path.join(self.dataset_dir, x), list_train)]
        # 排序
        list_test=sorted(list_test)
        list_train=sorted(list_train)
        # 保存训练集和测试集为json文件
        os.makedirs(self.output_folder, exist_ok=True)

        with open(os.path.join(self.output_folder, 'train_images.json'), 'w',encoding='utf-8') as j:
            json.dump(list_train, j, ensure_ascii=False, indent=4,
                      separators=(',', ':'))

        with open(os.path.join(self.output_folder, 'test_images.json'), 'w',encoding='utf-8') as j:
            json.dump(list_test, j, ensure_ascii=False, indent=4,
                      separators=(',', ':'))
        return list_train ,list_test,len(list_train),len(list_test)


if __name__ == '__main__':
    file_dir="E:/dataSets/underWater/typeI_data/underwater_typeI/DAVT训练集/UIEB/underwater/"
    output_folder="data05"
    class1=DataCollector(file_dir,output_folder)
    for img_name in class1.test_datasets:
        print(img_name)
        img = Image.open(img_name).convert('RGB')
        plt.imshow(img)
        plt.show()
        # print(label, img_name)
        break