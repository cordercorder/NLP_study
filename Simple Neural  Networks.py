import numpy as np
import math


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


class BPNetWork:

    def __init__(self,input_vec,hidden_vec,output_vec,learningrate,activator):
        '''
        :param input_vec: 输入层节点数
        :param hidden_vec: 隐藏层节点数(支持多个隐藏层)，但隐藏层至少2个
        :param output_vec: 输出层节点数
        :param learningrate: 学习率
        :param activator: 激活函数，可以自己选择
        '''
        self.input_vec=input_vec
        self.hidden_vec=hidden_vec
        self.output_vec=output_vec
        self.activator=activator
        self.learningrate=learningrate
        self.w=[]
        last_vec=input_vec
        for i in range(len(hidden_vec)):
            self.w.append(np.random.normal(0.0,1.0/pow(hidden_vec[i],0.5),(hidden_vec[i],last_vec)))
            last_vec=hidden_vec[i]

        self.w.append(np.random.normal(0.0,1.0/pow(output_vec,0.5),(output_vec,last_vec)))

    def UpdateWeightStochastic(self,input_data,labels):# 随机梯度下降
        input_data=np.array(input_data,ndmin=2).T
        labels=np.array(labels,ndmin=2).T
        output=[]
        output.append(self.activator(np.dot(self.w[0],input_data)))
        for i in range(1,len(self.w)-1):
            output.append(self.activator(np.dot(self.w[i],output[-1])))
        output.append(np.dot(self.w[-1],output[-1])) # 最后一层不加激活函数

        error=[]
        error.append(labels-output[-1])

        for i in range(len(self.w)-1,0,-1):
            error.append(np.dot(self.w[i].T,error[-1]))

        j=0
        for i in range(len(self.w)-1,0,-1):
            if i==(len(self.w)-1):
                self.w[i]=self.w[i]+self.learningrate*np.dot(error[j],output[i-1].T)
            else:
                self.w[i]=self.w[i]+self.learningrate*(np.dot(error[j]*output[i]*(1.0-output[i]),output[i-1].T))
            j+=1
        self.w[0]=self.w[0]+self.learningrate*(np.dot(error[j]*output[0]*(1.0-output[0]),input_data.T))

    def trainStochastic(self,data,labels,cnt):
        for i in range(len(data)):
            for j in range(cnt):
                self.UpdateWeightStochastic(data[i],labels[i])

    def UpdateWeightBatch(self,input_data,labels,sum_w):
        input_data=np.array(input_data,ndmin=2).T
        labels=np.array(labels,ndmin=2).T

        output=[]
        output.append(self.activator(np.dot(self.w[0],input_data)))

        for i in range(1,len(self.w)-1):
            output.append(self.activator(np.dot(self.w[i],output[-1])))

        output.append(np.dot(self.w[-1],output[-1]))

        error=[]
        error.append(labels-output[-1])

        for i in range(len(self.w)-1,0,-1):
            error.append(np.dot(self.w[i].T,error[-1]))

        j=0

        for i in range(len(self.w)-1,0,-1):
            if i==(len(self.w)-1):
                sum_w[i]=sum_w[i]+self.learningrate*np.dot(error[j],output[i-1].T)
            else:
                sum_w[i]=sum_w[i]+self.learningrate*np.dot(error[j]*output[i]*(1.0-output[i]),output[i-1].T)
            j+=1
        sum_w[0]=sum_w[0]+self.learningrate*np.dot(error[j]*output[0]*(1.0-output[0]),input_data.T)

    def trainBatch(self,data,labels,cnt):

        for i in range(cnt):
            sum_w=[]
            for item in self.w:
                sum_w.append(np.zeros(item.shape))
            for j in range(len(data)):
                self.UpdateWeightBatch(data[j],labels[j],sum_w)

            for j in range(len(self.w)):
                self.w[j]=self.w[j]+sum_w[j]/len(self.w)

    def predict(self,input_data):
        input_data=np.array(input_data,ndmin=2).T
        output=self.activator(np.dot(self.w[0],input_data))
        for i in range(1,len(self.w)-1):
            output=self.activator(np.dot(self.w[i],output))
        output=np.dot(self.w[-1],output)
        return output


if __name__=='__main__':
    N=BPNetWork(1,(20,20),1,0.01,sigmoid)
    j=-3.14
    data=[]
    labels=[]
    for i in range(10):
        data.append(j)
        labels.append(math.sin(j))
        j+=0.02
    N.trainBatch(data,labels,100000)
    for i in range(len(data)):
        print('data== %.6f , predict is %.6f , actual is %.6f'%(data[i],N.predict(data[i]),labels[i]))