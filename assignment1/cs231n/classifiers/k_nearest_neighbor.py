import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    dists1 = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        #print ('print Xtest', X[i,:].shape)
        #print ('print Xtrain', self.X_train[j].shape)
#         print ('Xtrain.Shape',self.X_train.shape)
#         print ('Xtest.Shape',X.shape)
        #dists1[i,j]=np.linalg.norm((X[i,:]-self.X_train[j,:]));
        test=X[i,:]-self.X_train[j,:]
#         print ('test.shape', test.shape)
        dists[i,j]=np.linalg.norm(test);
        #dists1[i,j]=np.sqrt(np.sum(test**2))
#         print ('print dist_result', dists[i,j])
#         print ('print dist1_result', dists1[i,j])
        
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      pass
#       print ('testdata.shape', X.shape)
      test=X[i,:]-self.X_train #500x3072 -5000x3072
    
#       print ('test.shape', test.shape)
      #dists[i,:]=np.linalg.norm(test);
      dists[i,:]=np.sqrt(np.sum((test)**2,axis=1))
      #print ('dist', dists[i,:].shape)
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    # Sum(xtest_i - ytrain_j) ** 2 = Sum3072pixel xtest_i ** 2 + Sum ytrain_j ** 2 - Sum 2 * xtest_i * ytrain_j

    X_square_sum = np.sum(X ** 2, axis=1) #500   Sum xtest_i ** 2 =  x1**2+....x3072**2   i =[1..500]
    X_train_square_sum = np.sum(self.X_train ** 2, axis=1) # 5000   Sum ytrain_j=  y1**2+....y3072**2 [j= 1..5000]
    X_mul_X_train = np.dot(X, self.X_train.T) #500x5000    Sum xtest_i * ytrain_j=  xtest_i1*ytrain_j1+....+xtest_i3072*ytrain_j3072
    #print('X2:',X_square_sum.shape, 'Xtrain_shape:',X_train_square_sum.shape,'XtrainXtes_shape',X_mul_X_train.shape)
        
    X_square_sum=np.reshape(X_square_sum, (-1, 1))# 500x1 Sum xtest_i ** 2
    X_train_square_sum=np.reshape(X_train_square_sum, (1, -1)) #1x5000  Sum ytrain_j ** 2
    
    temp=X_square_sum+X_train_square_sum #500x5000 = [i,j]    Sum xtest_i ** 2 + Sum ytrain_j ** 2
    
    #temp=X_square_sum[:,np.newaxis] + X_train_square_sum # [500,1] +[1,5000] = 500x5000
    dists = np.sqrt(temp - 2 * X_mul_X_train) # 5500x5000 - 500x5000= Sum xtest_i ** 2 + Sum ytrain_j ** 2 - Sum 2 * xtest_i * ytrain_j
#     print('X_square_sum', X_square_sum[:,np.newaxis].shape)
#     print('X_train_square_sum', X_train_square_sum.shape)
#     print ('dist.shapre',dists.shape)
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
#     labels = np.zeros((num_test, num_train))
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
         
      index=np.argsort(dists[i, :])# dist: 500x3072
#       print('shape of index',index.shape)   
      for n in range(k):
#             print (n,'index n:',index[n])
#             print ('shape of self_y_train', self.y_train.shape)
            label_nthbest_traindata=self.y_train[index[n]]
#             print ('label of data:', label_nthbest_traindata)
            closest_y.append(label_nthbest_traindata)
#             print('labels of k nearest',closest_y[n])
         
        
     
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      pass
#       print('labels of k nearest',closest_y)
      counts = np.bincount(closest_y)
#       print ('most common label',np.argmax(counts))
      y_pred[i]=np.argmax(counts)
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

