import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train= X.shape[0]
  num_class = W.shape[1]
#   softmax= np.zeros(num_train)
  loss_train_example=0.0
  dscores=0.0
  for order in range(num_train) :
    score=X[order,:].dot(W)#1x10= [1x3073] x [3073x10]    
    #numerical stability: http://cs231n.github.io/linear-classify/#softmax
    maxf= np.amax(score)   
    score-= maxf
    
    exp_score= np.exp(score)
#     print ('exp_score shape', exp_score.shape)
    softmax = exp_score/np.sum(exp_score) # 10     
    loss_train_example+=-np.log(softmax[y[order]])
# Gradient computing: http://cs231n.github.io/neural-networks-case-study/#grad
# Mathematical derivation: https://math.stackexchange.com/a/945918/359714
# summary softmax gradient final results: https://mlxai.github.io/2017/01/09/implementing-softmax-classifier-with-vectorized-operations.html
    # gradient
    dscores_onesample=softmax
#     print ('dscores 1 sampe ', dscores)
    dscores_onesample[y[order]]-=1
#     print ('dscores 1 sampe after - correct class ', dscores)
    temp=X[order,:].T
#     print ('temp shape', temp.shape)
    dW+=temp[:,None]*dscores_onesample
    
  data_loss=loss_train_example/num_train  
  reg_loss= 0.5*reg*np.sum(W*W)  
  loss= data_loss + reg_loss
  dW/=num_train
  dW+= reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train= X.shape[0]
  num_class = W.shape[1]
#   softmax= np.zeros(num_train)
  loss_train_example=0.0
  dscores=0.0
  score=X.dot(W) # [Nx3073]x[3073x10] = Nx10 = NxC
  maxf= np.amax(score,axis =1) #N
#   print('score shape', score.shape)
#   print('maxf shape', maxf.shape)
  score= (score.T-maxf).T  #NxC
#   print('score shape', score.shape)
  exp_score= np.exp(score)# NxC 
  softmax = (exp_score.T/np.sum(exp_score, axis=1)).T#NxC=(NxC.T/N).T, check xem moi hang co chia cho 1 gia tri tuong ung ko
#   print('softmax shape', softmax.shape)
  Allloss=softmax[np.arange(num_train),y] # N : chi lay loss o vi tri correct class
  Allloss= -np.log(Allloss)
# Calculate Gradient
  dscore= softmax #NxC
  dscore[np.arange(num_train), y]-=1 #NxC
  
  dW= X.T.dot(dscore)
  dW/=num_train
  dW+= reg*W
 
  loss= np.sum(Allloss)/num_train
  loss+=0.5*reg*np.sum(W*W) 

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

