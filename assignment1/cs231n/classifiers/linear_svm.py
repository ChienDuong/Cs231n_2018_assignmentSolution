import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
#     print ('X[i] shape:',X[i].shape)    
#     print ('W shape:', W.shape)
    scores = X[i].dot(W)
#       compute gradient, after forwardpass
#http://cs231n.github.io/optimization-1/#vis
#     print ('score shape:', scores.shape)
    correct_class_score = scores[y[i]]
  
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j]+=X[i,:]#3073
        dW[:,y[i]]+=-X[i,:]
      

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW /=num_train
  # Add regularization to the gradient.
  dW+= reg*2*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
#   print ('X shape', X.shape)
#   print ('y shape', y.shape)
#   print ('W shape', W.shape)
  # X : 500x3073; y= 500, W= 3073x10
  scores = X.dot(W) #500x10
#   print ('score shape:', scores.shape)
#   print ('y value:',y)
  correct_class_score = np.choose(y, scores.T)#500  correct_class_score = np.choose(y, scores.T)#500

#   print ('correct_class_score shape:', correct_class_score.shape) 
#   temp=scores.T - correct_class_score
#   print ('temp shape:',temp.shape)
  margins= np.maximum(0,(scores.T- correct_class_score).T +1)
#   print ('margins shape:', margins) #500,10 

# set margins của correct labels= 0, previous, loss of correct labesl = margins =1
#https://stackoverflow.com/questions/23435782/numpy-selecting-specific-column-index-per-row-by-using-a-list-of-indexes/23435843?fbclid=IwAR0p5KAlnjQ_0YyH-J-B0UzYr24jjrMTTl22ccwkyBTzV3HLOkdtTY641bc#23435843
  margins[np.arange(X.shape[0]),y]=0
#   print ('margins shape:', margins) #500,10
  loss= np.sum(margins)
  loss /= y.shape[0]

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
#Honor code wrong
#   temp= np.sum(margins, axis=1)
#   print ('temp shape', temp.shape)
#     #Set gradient for correct class
#   index_bigger_zero=np.argwhere(temp > 0)  
#   index_bigger_zero=index_bigger_zero.flatten()  
#   print ('sum margin bigger than 0', index_bigger_zero.shape)
#   print ('dw shape', dW.shape)
#   temp=X[index_bigger_zero]
#   print ('temp shape:', np.sum(temp, axis=0).shape)
#   print ('dw shape', dW.shape)
#   dW[:,y[index_bigger_zero]]=-np.sum(X[index_bigger_zero],axis=0)

#     # Set gradient for incorrect class
#   dW/= y.shape[0]
#   dW+= reg*2*W

# reference code: awesome
# https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html
  
  binary = margins
  binary[margins > 0] = 1 #500x10
#   print ('binaryshape',binary)

  row_sum = np.sum(binary, axis=1)
#   print ('row_sum',row_sum.shape)#500
  num_train=X.shape[0]
  binary[np.arange(num_train), y] = -row_sum.T
#   print ('binaryshape',binary)
#   print('X shape', X.shape)# 500x3073
#   print('binary shape', binary.shape)# 500x10
  dW = np.dot(X.T, binary) # 3073x10
  dW/= y.shape[0]
  dW+= reg*2*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
