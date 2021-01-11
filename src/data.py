import numpy as np
import pandas as pd


def data_process(df, user_col, item_col, rating_col, test_ratio = 0.25, shuffle=True):
    '''
        Make ratings data to matrxi users versus items 
        It returs the 3 types of data with same size n_users x n_itmes 
        It's not memory efficient way
    '''
    n_data = len(df)
    
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    
    data = np.zeros((n_users, n_items))
    mask_data = np.zeros((n_users, n_items))

    train_data = np.zeros((n_users, n_items))
    train_mask_data = np.zeros((n_users, n_items))
    
    test_data = np.zeros((n_users, n_items))
    test_mask_data = np.zeros((n_users, n_items))

    if shuffle:
        random_idx = np.random.permutation(n_data)
        train_idx = random_idx[0:int(n_data * test_ratio)]
        test_idx = random_idx[int(n_data * test_ratio):]
    else:
        train_idx = n_data.index[0:int(n_data * test_ratio)]
        test_idx = n_data.index[int(n_data * test_ratio):]
        
    users = df[user_col]
    items = df[item_col]
    ratings = df[rating_col]
    
    for idx, (user, item, rating) in enumerate(zip(users, items, ratings)):
        data[user, item] = rating
        mask_data[user, item] = 1
        if idx in train_idx:
            train_data[user, item] = rating
            train_mask_data[user, item] = 1
        
        if idx in test_idx:
            test_data[user, item] = rating
            test_mask_data[user, item] = 1
        
    
    return data, mask_data, train_data, train_mask_data, test_data, test_mask_data

