import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class MultiLayerConvNet(object):
  """
  A multi-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax

and:
(1) [conv-relu-pool]*N-conv-relu-affine*M-softmax
(2) [conv-relu-pool]*N-affine*M-softmax
(3) [conv-relu-conv-relu-pool]*N-affine*M-softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, num_filters, hidden_dims, cnn_arch,
                input_dim=(3, 32, 32), filter_size=7,
                num_classes=10, weight_scale=1e-3, reg=0.0, dropout=0,
               dtype=np.float32,use_batchnorm=False,use_Xavier=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.fw=filter_size
    self.cnn_arch=cnn_arch
    self.use_bn=use_batchnorm
    self.use_Xavier=use_Xavier
    self.use_dropout = dropout > 0
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C,H,W=input_dim
    F, fw=[C]+num_filters, filter_size
    N,M=len(num_filters),len(hidden_dims)
    self.N,self.M=N,M
    strw, padw=1, (filter_size-1)/2
    if self.use_Xavier:
      Wconvs={'W'+str(i+1):np.random.randn(F[i+1],F[i],fw,fw)*np.sqrt(2.0/F[i]/fw/fw) for i in xrange(N)}
    else:
      Wconvs={'W'+str(i+1):np.random.randn(F[i+1],F[i],fw,fw)*weight_scale for i in xrange(N)}
    bconvs={'b'+str(i+1):np.zeros(F[i+1]) for i in xrange(N)}
    if cnn_arch==1:
      pool_num=N-1
    if cnn_arch==2:
      pool_num=N
    if cnn_arch==3:
      pool_num=N/2
    D=[F[N]*input_dim[1]*input_dim[2]/(4**pool_num)]+hidden_dims
    #print 'first hidden size F: %d'% (F[N]*input_dim[1]*input_dim[2]/(4**pool_num))
    #print 'fn:%d'
    #print 'F:',F
    #print 'D:',D
    if self.use_Xavier:
      Waffs={'W'+str(i+1+N):np.random.randn(D[i],D[i+1])*np.sqrt(2.0/D[i]) for i in xrange(M)}
    else:
      Waffs={'W'+str(i+1+N):np.random.randn(D[i],D[i+1])*weight_scale for i in xrange(M)}
    baffs={'b'+str(i+1+N):np.zeros(D[i+1]) for i in xrange(M)}
    self.params.update(Wconvs)
    self.params.update(bconvs)
    self.params.update(Waffs)
    self.params.update(baffs)
    if self.use_bn:
      gamconvs={'gamma'+str(i+1):np.ones(F[i+1]) for i in xrange(N)}
      betaconvs={'beta'+str(i+1):np.zeros(F[i+1]) for i in xrange(N)}
      gamaffs={'gamma'+str(i+1+N):np.ones(D[i+1]) for i in xrange(M)}
      betaaffs={'beta'+str(i+1+N):np.zeros(D[i+1]) for i in xrange(M)}
      self.params.update(gamconvs)
      self.params.update(betaconvs)
      self.params.update(gamaffs)
      self.params.update(betaaffs)
    if self.use_Xavier:
      self.params['W'+str(M+N+1)]=np.random.randn(D[M],num_classes)*np.sqrt(2.0/D[M])
    else:
      self.params['W'+str(M+N+1)]=np.random.randn(D[M],num_classes)*weight_scale
    self.params['b'+str(M+N+1)]=np.zeros(num_classes)
    self.bn_params = []
    if self.use_bn:
      self.bn_params = [{'mode': 'train'} for i in xrange(M+N)]
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = self.fw
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_bn:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    N,M,cnn_arch=self.N,self.M,self.cnn_arch
    fwd_data={}
    #print "original Xshape:",X.shape
    for i in xrange(N):
      convW,convb=self.params['W'+str(i+1)], self.params['b'+str(i+1)]
      if self.use_bn:
        gam,beta=self.params['gamma'+str(i+1)],self.params['beta'+str(i+1)]
        X,convcache=conv_bn_relu_forward(X,convW,convb,conv_param,gam,beta,self.bn_params[i])
      else:
        X,convcache=conv_relu_forward(X,convW,convb,conv_param)
      fwd_data['conv'+str(i+1)]=convcache
      if self.use_dropout:
        X,dropcache=dropout_forward(X,self.dropout_param)
        fwd_data['drop'+str(i+1)]=dropcache
      if cnn_arch==2 or cnn_arch==1 and i!=N-1 or cnn_arch==3 and i&1!=0:
        #print 'pooled!'
        X,poolcache=max_pool_forward_fast(X,pool_param)
        fwd_data['pool'+str(i+1)]=poolcache
      #print 'conv forward i:%d Xshape:'%i,X.shape
    for i in range(M):
      affW,affb=self.params['W'+str(i+1+N)], self.params['b'+str(i+1+N)]
      if self.use_bn:
        gam,beta=self.params['gamma'+str(i+1+N)],self.params['beta'+str(i+1+N)]
        X,affcache=affine_bn_relu_forward(X,affW,affb,gam,beta,self.bn_params[i+N])
      else:
        X,affcache=affine_relu_forward(X,affW,affb)
      fwd_data['aff'+str(i+1+N)]=affcache
      if self.use_dropout:
        X,dropcache=dropout_forward(X,self.dropout_param)
        fwd_data['drop'+str(i+1+N)]=dropcache
    scoreW,scoreb=self.params['W'+str(M+N+1)], self.params['b'+str(M+N+1)]
    scores,scorecache=affine_forward(X,scoreW,scoreb)
    fwd_data['score']=scorecache
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    reg=self.reg
    loss,dscores=softmax_loss(scores,y)
    dout,dscoreW,dscoreb=affine_backward(dscores,fwd_data['score'])
    grads['W'+str(N+M+1)],grads['b'+str(N+M+1)]=dscoreW,dscoreb
    for i in xrange(M-1,-1,-1):
      if self.use_dropout:
        dout=dropout_backward(dout,fwd_data['drop'+str(i+1+N)])
      if self.use_bn:
        dout,daffW,daffb,dgam,dbeta=affine_bn_relu_backward(dout,fwd_data['aff'+str(i+1+N)])
        grads['gamma'+str(i+1+N)],grads['beta'+str(i+1+N)]=dgam,dbeta
      else:
        dout,daffW,daffb=affine_relu_backward(dout,fwd_data['aff'+str(i+1+N)])
      loss+=0.5*reg*np.sum(self.params['W'+str(i+1+N)]**2)
      daffW+=reg*self.params['W'+str(i+1+N)]
      grads['W'+str(i+1+N)],grads['b'+str(i+1+N)]=daffW,daffb
    for i in xrange(N-1,-1,-1):
      if cnn_arch==2 or cnn_arch==1 and i!=N-1 or cnn_arch==3 and i&1!=0:
        dout=max_pool_backward_fast(dout,fwd_data['pool'+str(i+1)])
      if self.use_dropout:
        dout=dropout_backward(dout,fwd_data['drop'+str(i+1)])
      if self.use_bn:
        dout,dconvW,dconvb,dgam,dbeta=conv_bn_relu_backward(dout,fwd_data['conv'+str(i+1)])
        grads['gamma'+str(i+1)],grads['beta'+str(i+1)]=dgam,dbeta
      else:
        dout,dconvW,dconvb=conv_relu_backward(dout,fwd_data['conv'+str(i+1)])
      loss+=0.5*reg*np.sum(self.params['W'+str(i+1)]**2)
      dconvW+=reg*self.params['W'+str(i+1)]
      grads['W'+str(i+1)],grads['b'+str(i+1)]=dconvW,dconvb
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
