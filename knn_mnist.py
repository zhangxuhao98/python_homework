# -*- coding: utf-8 -*-

#计算列表中出现次数最多的元素，并返回该元素
def countMax(LIST):
    number={}
    max_key=None
    for a in LIST:
        if a not in number:
            # 计算该元素在列表中出现的次数
            count = LIST.count(a)
            # 保存到字典中
            number[a] = count
            # 记录最大元素
            if count > number.get(max_key, 0):
                max_key = a
    return max_key

#计算准确率
def getAccuracy(testtarget,predictions):  
    correct = 0
    name=input("请输入用于存放预测数据的文件名：")
    for x in range(len(testtarget)):  
        if testtarget[x] == predictions[x]:
            correct+=1
        with open(name,"a") as f:
            f.write(">predicted = " + repr(predictions[x]) + ",actual = " + repr(testtarget[x])+"\n")
    print("文件创建成功！文件名为"+name+"!")
        
        
    return (correct/float(len(testtarget))) * 100.0

def KNN(traindata,testdata,traintarget):
    import numpy as np
    k = int(input("请输入'k':"))
    pre=[]
    TrainSize=traindata.shape[0]
    for inX in testdata:
        
        diffMat=np.tile(inX,(TrainSize,1))-traindata#计算欧氏距离
        sqDiffMat=diffMat**2
        sqDistance=sqDiffMat.sum(axis=1)
        distance=sqDistance**0.5                    #
        sortedIndex=distance.argsort()#从小到大排序后，返回值对应的索引
        kCount=[]
        for i in range (k): #统计前k个数据类的数量
            label = traintarget[sortedIndex[i]]#找到索引对应的标签
            kCount.append(label)
            
        pre.append(countMax(kCount))
    return [pre,countMax(kCount)]

   
def main():
    # 导入数据集
    from sklearn.datasets import load_digits
    mnist = load_digits()
    X = mnist.data
    y = mnist.target

    '''
    分割数据集
    使用train_test_split，利用随机种子random_state及stratify在每一个target采样25%的数据作为测试集。
    经过测试，使用stratify后，当k=80时，准确率仍在80%以上，而不使用时为50%
    '''
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=33,stratify=y)
    #,stratify=y
    #判断运行测试集，或者自行输入数据测试

    pre=KNN(X_train,X_test,y_train)[0]
        
    print(getAccuracy(y_test.tolist(),pre))#将numpy.ndarray转换为list，不转好像也不报错
    from sklearn.metrics import accuracy_score
        
if __name__ == '__main__':
    main()
