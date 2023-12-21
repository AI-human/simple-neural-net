import numpy as np

class ActivationFunction:
    def __init__(self) -> None:
      pass

    def sigmoid(self,z: int):
      return 1/(1+np.exp(-z))

class Nerual_Net:
  def __init__(self):
    pass
  def layer(self,a_inp,W,b,activation_function): # layer is dense in tensorflow

    """
    Computes dense layer
    a_inp (ndarray (n, )) = data
    W  (ndarray (n,j)) = Weight matrix, n features per unit, j units
    b  (ndarray (j, )) = bias vector, j units  
    activation_function = eg. sigmoid,relu 

    a_out = activation_function( (w*x)+b)
    """
    neurons = W.shape[1]
    a_out = np.zeros(neurons) # initally zeros
    for j in range(neurons): 
      w = W[:,j] # every jth column
      z = np.dot(w,a_inp) + b # np.dot uses multiple core to computes in parallel 
      a_out[j] = activation_function(z)
    return a_out
  
  def sequential(self,x,W1,b1,W2,b2): # max five layer 
    a1 = self.layer(x,W1,b1,ActivationFunction.sigmoid)
    a2 = self.layer(x,W2,b2,ActivationFunction.sigmoid)
    return a2

  def predict(self,X,W1,b1,W2,b2):
    m = X.shape[0]
    p = np.zeros(m)
    for i in range(m):
      p[i,0] = self.sequential(X[i],W1,b1,W2,b2)
    return p