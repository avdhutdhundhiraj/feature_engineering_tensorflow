#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### feature engineering functions ########
def catag_feature_vocab(cat_cols,df):
    feature_cols = [tf.feature_column.categorical_column_with_vocabulary_list(x, list(df[x].unique())) for x in cat_cols]
    return feature_cols

def numerical_feature_cols(num_cols,df):
    feature_cols = [tf.feature_column.numeric_column(x, dtype=tf.float32) for x in num_cols]
    return feature_cols

def get_gen_cat_list(df1):
    df_types = df1.dtypes.reset_index(name = 'type')
    df_types['col']=df_types['index']
    return list(df_types[df_types['type']=='object']['col'])

def df_to_dataset(df,target_col, shuffle=True, batch_size=32):
    dataframe = df.copy()
    labels = dataframe.pop(target_col)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
    return ds


# In[ ]:




