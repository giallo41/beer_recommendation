
import numpy as np

from keras.layers import Input, Embedding, Dot, Flatten
from keras.regularizers import *
from keras.models import *
import keras.backend as K
import keras





class EmbedModel():
    
    def __init__(self, n_users, n_items, embed_size=20, n_l2=1e-5):
        
        self.n_users = n_users
        self.n_items = n_items
        self.embed_size = embed_size
        self.n_l2 = n_l2
        K.clear_session()
    
    def build(self):
        users = Input(shape=(1,), dtype='int64', name='user_input')
        self.u_embed = Embedding(self.n_users, 
                                self.embed_size, 
                                input_length = 1,
                                embeddings_regularizer = l2(self.n_l2),
                                name='user_embed')(users)

        items = Input(shape=(1,), dtype='int64', name='item_input')
        self.i_embed = Embedding(self.n_items, 
                                self.embed_size, 
                                input_length = 1,
                                embeddings_regularizer = l2(self.n_l2),
                                name='item_embed')(items)

        x = Dot(axes=2)([self.u_embed,self.i_embed])
        x = Flatten()(x)

        self.model = Model([users, items], x)
        return self.model
    
    def get_recommendation(self, user_id, item_idx_dic, top_n = 10):
        user_embed_weight = self.model.get_layer(name='user_embed').get_weights()[0]
        item_embed_weight = self.model.get_layer(name='item_embed').get_weights()[0]
        item_scores = np.dot(user_embed_weight[user_id],item_embed_weight.T)
        
        top_scores_idx = np.argpartition(item_scores, -top_n)[-top_n:][::-1]
        
        rtn = {}
        for item in top_scores_idx:
            rtn[item_idx_dic[item]] = item_scores[item]
    
        return rtn