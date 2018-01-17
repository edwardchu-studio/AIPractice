import pandas as pd
# import keras as krs
import numpy as np
from sklearn.svm import SVC
class classifier:
    __doc__ = "a classifier wrapper of all classifiers, typically artificial neural networks like CNN, LSTM, etc."

    def __init__(self):
        self.alph='abcdefghijklmnopqrstuvwxyz?'

    def readRawData(self,dir,type=0):
        if type==0: # from csv
            _data=pd.read_csv(dir)
        elif type==1: # from excel
            _data=pd.read_excel(dir)
        else:
            _data=pd.DataFrame()

        #print(_data.head(5))

        self.data=_data.sample(frac=1).reset_index(drop=True)
        self.columns=list(self.data.columns.values)

    def prepareDataSet(self,ratio=0.8):
        '''
        :return:X,Y,x,y: traning X, training Y, testing x, testing y
        '''
        numRecords=self.data.size

        self.numTrain=int(numRecords*ratio)
        X=np.zeros((self.numTrain,len(self.columns)-1,),int)
        Y=np.zeros((self.numTrain),int)


        tX=np.zeros((numRecords-self.numTrain,len(self.columns)-1,),int)
        tY=np.zeros((numRecords-self.numTrain),int)

        for i,v in enumerate(self.data.head(self.numTrain)['class']):

            if v == 'e':
                Y[i]=0
            else:
                Y[i]=1
        for i,v in enumerate(self.data.tail(numRecords-self.numTrain)['class']):

            if v == 'e':
                tY[i]=0
            else:
                tY[i]=1




        for i,v in enumerate(self.data.head(self.numTrain).values):
            #print(v,type(v))
            for j,f in enumerate(v[1:]):
                #print(X[i,j])
                X[i,j]=self.alph.index(f)
                #print(X[i,j])

        for i,v in enumerate(self.data.head(numRecords-self.numTrain).values):
            #print(v,type(v))
            for j,f in enumerate(v[1:]):
                tX[i,j]=self.alph.index(f)


        # print(numRecords)

        print(X.shape,Y.shape,tX.shape,tY.shape)
        #print(X,Y,tX,tY)


        return X,Y,tX,tY

    def svc(self):
        '''
        :param X: training X
        :param Y: training Y
        :param x: testing x
        :param y: testing y
        :param test: boolean parameter to indicate test or not
        :return: classifier
        '''
        X,Y,tX,tY=self.prepareDataSet(ratio=0.5)
        self.classifier=SVC(probability=True)
        self.classifier.fit(X,Y)

        print(self.classifier.predict(tX[0].reshape(1,-1)),tY[0])

        print(self.classifier.score(tX,tY))

    def cnn(self,X,Y,x,y,test=False):
        '''
        :param X: training X
        :param Y: training Y
        :param x: testing x
        :param y: testing y
        :param test: boolean parameter to indicate test or not
        :return: cnn classifier
        '''
        pass



def main():
    c=classifier()
    c.readRawData('../data/mushrooms.csv',0)
    c.svc()

if __name__ == '__main__':
    main()
