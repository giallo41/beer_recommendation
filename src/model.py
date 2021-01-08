

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
                                embeddings_regularizer = l2(self.n_l2))(users)

        items = Input(shape=(1,), dtype='int64', name='item_input')
        self.i_embed = Embedding(self.n_items, 
                                self.embed_size, 
                                input_length = 1,
                                embeddings_regularizer = l2(self.n_l2))(items)

        x = Dot(axes=2)([self.u_embed,self.i_embed])
        x = Flatten()(x)

        self.model = Model([users, items], x)
        return self.model
    
 #   def compile(self, optimizer = 'Adam', loss='mse', **kwargs):
 #       self.model.compile(optimizer, loss)
        
 #   def fit(self, x_data, y_data, batch_size, epochs, validation_split=0.2, verbose=2):
 #       return self.model.fit(x=x_data, 
 #                              y=y_data, 
#                               batch_size=batch_size,
#                               epochs=epochs, 
#                               validation_split=validation_split, 
#                               verbose=verbose)