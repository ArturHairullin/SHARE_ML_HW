import numpy as np

class MyBernoulliNBClassifier():
    def __init__(self, priors=None):
        pass
    @staticmethod
    def p(y,i):
        k=0
        for a in y:
            if a==i:
                k+=1
        return k/len(y)
    
    def maxtrue(self,X,y,i):
        s = [0.0]*len(X[0])
        for j in range(len(y)):
            if y[j]==i:
                for k in range(len(X[j])):
                    s[k]+=X[j][k]
        for j in range(len(s)):
            s[j]=s[j]/(len(y)*self.p(y,i))
        return s
    
    def r(self,x,i):
        tmp = self.p_y[i] 
        for j in range(len(x)):
            tmp*=(self.s[i][j]**x[j])*((1-self.s[i][j])**(1-x[j]))
        return tmp
    
    def fit(self, X, y):
        self.p_y = []
        self.s = []
        self.ans = []
        for i in set(y):
            self.p_y.append(self.p(y,i))
            self.s.append(self.maxtrue(X,y,i))
            self.ans.append(i)
        #print(self.p_y)
        #print(self.s)
        #print(self.ans)
    
    def predict(self, X):
        result = []
        for x in X:
            max = 0
            val = self.ans[0]
            for i in range(len(self.ans)):
                if self.r(x,i)>max:
                    max = self.r(x,i)
                    val = self.ans[i]
            result.append(val)
        return np.array(result)              
    
    def predict_proba(self, X):
        result = []
            
        for x in X:
            p = []
            c = 0.0
            for i in range(len(self.ans)):
                c+=self.r(x,i)
            for i in range(len(self.ans)):
                p.append(self.r(x,i)/c)
            result.append(np.array(p))
        return np.array(result)
    
    def score(self,X, y):
        p = self.predict(X)
        k = 0
        for i in range(len(y)):
            if y[i]==p[i]:
                k+=1
        return k/len(y)
