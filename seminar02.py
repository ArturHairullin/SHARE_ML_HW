import numpy as np
import matplotlib.pyplot as plt


def load_data(path='mnist.npz'): 
    with np.load(path, allow_pickle=True) as f:  # pylint: disable=unexpected-keyword-arg
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)
def short(x):
    a=[]
    k=0
    for i in range(len(x)-1):
        if x[i]!=0 and x[i]==x[i+1]:
            if k!=x[i]:
                a.append(x[i])
                k=x[i]
    return a
def outline_analyze(x):
    horizon = []
    vertical = []
    for i in range(len(x)) :
        k=0
        s=0
        for j in range(len(x[i])):
            if x[i][j]!=0 and k==0:
                s+=1
                k=1
            if x[i][j]==0:
                k=0
        horizon.append(s)
        
    for i in range(len(x)) :
        k=0
        s=0
        for j in range(len(x[i])):
            if x[j][i]!=0 and k==0:
                s+=1
                k=1
            if x[j][i]==0:
                k=0
        vertical.append(s)
        
    return [short(horizon),short(vertical)]


(x_train, y_train), (x_test, y_test) = load_data(path='mnist.npz')
template_list = []
for i in range(10):
    template_list.append([])

for i in range(20):
    template_list[y_train[i]].append(outline_analyze(x_train[i]))
    
class MyFirstClassifier(object):
    def __init__(self):
        pass
    def fit(self, x_train, y_train):
        pass
    def predict(self, x_test):
        result = []
        for i in x_test:
            val = np.random.randint(low=0, size=1, high=10)[0]
            for j in range(10):
                s = outline_analyze(i)
                if s in template_list[j]:
                    val = j
                    break
            result.append(val)
        return result
def accuracy_score(pred, gt):
    return np.mean(pred==gt)