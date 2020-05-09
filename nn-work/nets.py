import numpy as np
class Net():
    def predict(self,x):
        return self.activate(self, np.dot(x, self.W))
    def activate(self, v_i):
        if v_i >= 0:
            return 1
        elif v_i<0:
            return -1
    activate = np.vectorize(activate)
    def savesweights(self, fname):
        np.save('%s.npy'% fname, self.W)
    def loadweights(self, fname):
        self.W= np.load('%s.npy'%fname)

class Hebb(Net):
    def fit(self, x, z, max_epochs= 50):
        self.W = np.zeros(x.shape[0] - 1)
        for i in range(max_epochs):
            for input_,target in zip(x,z):
                v_i = np.dot(input_,self.W)
                z_i = self.activate(self,v_i)
                self.W= self.W + input_ * target
        return self.W
class Perceptron(Net):
    def predict(self,x):
        return self.activate(self, np.dot(x, self.W),self.theta)
    def fit(self,x,z,alpha=1, theta=1):
        self.W = np.random.random(x.shape[0]-1)
        self.W_old =np.zeros(x.shape[0]-1)
        self.theta = theta
        conv_epoch =0
        while True:
            for input_, target in zip(x,z):
                v_i = np.dot(input_,self.W)
                z_i = self.activate(self, v_i,theta)
                if z_i != target:
                    self.W= self.W_old + alpha * target* input_
            conv_epoch += 1
            if np.max(np.abs(self.W - self.W_old))==0:
                break
            else:
                self.W_old = self.W
        return self.W, conv_epoch
    def activate(self, v_i, theta):
        if v_i > theta:
            return 1
        elif -theta <= v_i and  v_i <= theta:
            return 0
        elif v_i < -theta:
            return -1
    activate = np.vectorize(activate)
class Adaline(Net):
    def fit(self, x, z, alpha=1, tolerance= 1e-6):
        self.W = np.random.random(x.shape[0]-1)
        self.W_old =np.zeros(x.shape[0]-1)
        conv_epoch =0
        while True:
            for input_, target in zip(x,z):
                v_i = np.dot(input_,self.W)
                z_i = self.activate(self, v_i)
                if target != z_i:
                    self.W= self.W_old + alpha* (target - z_i) * input_
            conv_epoch += 1
            if np.max(np.abs(self.W - self.W_old))<=tolerance:
                break
            else:
                self.W_old = self.W
        return self.W, conv_epoch 
def getR(x):
    return np.sum(np.dot(x.T,x),axis=0)/len(x)
if __name__ ==  '__main__':
    x_train = np.array([
        [-1, -1, 1],
        [-1,  1, 1],
        [ 1, -1, 1],
        [ 1,  1, 1]
    ])
    y_train = np.array([
        -1,-1,-1,1
    ])
    print("------Hebb-------")
    HebbNet = Hebb()
    HebbNet.fit(x_train,y_train,10)
    y_pred = HebbNet.predict(x_train)
    print(y_train)
    print(y_pred)
    HebbNet.savesweights("weights_hebb")
    HebbNet1 = Hebb()
    HebbNet1.loadweights("weights_hebb")
    y_pred = HebbNet1.predict(x_train)
    print(y_train)
    print(y_pred)
    print("-----Perceptron------")
    PerceptronNet = Perceptron()
    alphas = [0.1 * i for i in range(1,11)]
    theta = 1
    for alpha in alphas:
        _,converged = PerceptronNet.fit(x_train, y_train,alpha,theta)
        y_pred = PerceptronNet.predict(x_train)
        print("a=",alpha,"e=",converged)
        print(y_pred)
        print(y_train)
        
    print("-----Adaline------")
    AdalineNet = Adaline()
    R = getR(x_train[:,:2])
    alpha = np.max(R)/2
    print("a=",alpha)
    _, converged = AdalineNet.fit(x_train,y_train, alpha=alpha)
    y_pred = AdalineNet.predict(x_train)
    print("e=",converged)
    print(y_pred)
    print(y_train)