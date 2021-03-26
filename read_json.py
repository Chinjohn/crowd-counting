# -*- coding: utf-8 -*-
#!/usr/bin/python
import json
import os
import shutil
# f = open("test_VisDrone.json", 'r', encoding='utf-8')
# ln = 0
# line1 = []
# for line in f.readlines():
#     # print(line)
#     for i in line.split(','):
#         print(i)
#         line1.append(i)
#     dic = json.loads(line)
#
#     # t = dic['id'],dic['com_name'],dic['com_register_time']
#     # f = open("out/data.txt",'a',encoding='utf-8')
#     # f.writelines(str(t));f.write("\n")
# # f.write(str(ln))
# print(len(line1))
# # print(ln)
# f.close()





print ("+++++++++++++++++++++++++++++")
print ("文件移动")
print ("+++++++++++++++++++++++++++++")

#############################
#复制文件到指定的目录下
def copyFiles(sourceDir, targetDir, filename):
    # path, sfileName = os.path.split(filename)
    sourcePath = sourceDir + filename[0:5]+'/' + filename[5:]
    print(sourcePath)
    targetPath = targetDir + filename
    print(sourcePath)
    shutil.copyfile(sourcePath, targetPath)
    print("copy %s -> %s\n" % (sourcePath, targetPath))


#############################
#解析json文件
def loadFont(fileName):
    f = open(fileName)  #设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
    lists = json.load(f)
    return lists

def execute():
    basePath = "/media/r8/725EF1325EF0F029/Graduation_project/CSRNet-pytorch-master/"
    lists = loadFont(basePath+"test_VisDrone.json")
    for item in lists:
        # path = item['path']
        # path1 = item.split('/')[4]
        path1 = item.split('/')[4].replace('.jpg', '.h5')
        # path1 = item.split('/')[4]

        # print(path1)
        targetDir="/media/r8/725EF1325EF0F029/Graduation_project/CSRNet-pytorch-master/test/gt/"
        sourceDir="/media/r8/147007E07007C786/da si/diploma_project/VisDrone2020-CC/train/gt/"
        copyFiles(sourceDir, targetDir, path1)

if __name__=='__main__':
    execute()
    print("执行完成.")