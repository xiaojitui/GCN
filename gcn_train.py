#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from gcnmodel import GCNModel


# In[ ]:


'''
cora
7 classes
2708 papers
feature dim: 1433

cora.content: <paper_id> <word_attributes>+ <class_label> 
cora.cites: <ID of cited paper> <ID of citing paper>
'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


def train():
    # Start training and evaluation
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess)


# In[ ]:


## test

params = {}
params['dataset'] = 'cora'
params['gpuid'] = 0
params['name'] = 'test' # train, test, val
params['lr'] = 0.01
params['epochs'] = 100
params['l2'] = 5e-4
params['seed'] = 1
params['optimizer'] = 'adam'

params['gcn_dim'] = 16 
params['dropout'] = 0.5

params['restore'] = 'store_true'
params['log_dir'] = './log'
params['model_dir'] = './models/'
params['config_dir'] = './config/'


## random seed
tf.set_random_seed(params['seed'])
np.random.seed(params['seed'])

## gpu
set_gpu(params['gpuid'])


## load model
model = gcnmodel(params) # args are above params

## train
train()


# In[ ]:




