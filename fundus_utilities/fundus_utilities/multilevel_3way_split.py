# Warpper for multi-level-split from https://github.com/lmkoch/multi-level-split/

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SEED = 12345

def multilevel_3way_split(df:pd.DataFrame, proportions:list=[0.6, 0.2, 0.2], seed:int=SPLIT_SEED, stratify_by:str=None, split_by:str=None, verify_proportions_with_atol:float=None):
    """Split a dataframe into train, validation, and test sets. Options to split by group (i.e. keep groups together) and stratify by label.

    Args:
        df (pd.DataFrame): dataframe to split
        proportions (list, optional): Proportions for train, validation, and test sets. Defaults to [0.6, 0.2, 0.2].
        seed (int, optional): random seed. Defaults to 12345.
        stratify_by (str, optional): column to stratify by. Defaults to None.
        split_by (str, optional): column to group together and split by. Defaults to None.
        verify_proportions_with_atol (float, optional): If not None, verify that the returned proportions are within this tolerance to the `proportions` list. Defaults to None.

    Returns:
        pd.DataFrame: train set
        pd.DataFrame: validation set
        pd.DataFrame: test set
    """
    
    # Split proportions [train, val, test]
    p = proportions

    dev_df, test_df = multilevel_train_test_split(df, df.index, test_split=p[1], seed=seed, stratify_by=stratify_by, split_by=split_by)

    train_df, val_df = multilevel_train_test_split(dev_df, dev_df.index, test_split=p[2]/(p[0]+p[1]), seed=seed, stratify_by=stratify_by, split_by=split_by)

    if verify_proportions_with_atol is not None:
        assert np.isclose(len(train_df)/len(df), p[0], atol=verify_proportions_with_atol), f"fraction is not {p[0]} but is {len(train_df)/len(df)}"
        assert np.isclose(len(val_df)/len(df), p[1], atol=verify_proportions_with_atol), f"fraction is not {p[1]} but is {len(val_df)/len(df)}"
        assert np.isclose(len(test_df)/len(df), p[2], atol=verify_proportions_with_atol), f"fraction is not {p[2]} but is {len(test_df)/len(df)}"

    return train_df, val_df, test_df

# From https://github.com/lmkoch/multi-level-split/
def multilevel_train_test_split(df:pd.DataFrame, 
                                index,
                                split_by=None,
                                stratify_by=None,
                                test_split=0.1,
                                seed=None):
    """Split pandas dataframe into train and test splits. Options to split by
    group (i.e. keep groups together) and stratify by label.

    Args:
        df (pandas dataframe): dataframe to split. Must contain columns index as well as
                     split_by and stratify_by (if not None)
        index (string): name of column that acts as index to dataset
        split_by (str, optional): column to group together and split by. Defaults to None.
        stratify_by (str, optional): column to stratify by. Defaults to None.
        test_split (float, optional): test proportion. Defaults to 0.1.
        seed (int, optional): random seed can be fixed for reproducible splits. 
                              Defaults to None.
    """
    if (index == df.index).all():
        df["_index_"] = index
        index = "_index_"

    if index not in df:
        raise ValueError(f'{index} not in df')
        
    if split_by is None:
        split_by = index
    elif split_by not in df:
        raise ValueError(f'{split_by} not a column in df')

    df_unique = df.drop_duplicates(subset=[split_by])

    if stratify_by is None:
        stratify = None
    else:
        if stratify_by in df:
            stratify = df_unique[stratify_by]
        else:
            raise ValueError(f'{stratify_by} not a column in df')
 
    train_ids, test_ids = train_test_split(df_unique[split_by], 
                                           test_size=test_split,
                                           stratify=stratify,
                                           random_state=seed)
    
    df_train = df.set_index(split_by).drop(test_ids).reset_index()
    df_test = df.set_index(split_by).drop(train_ids).reset_index()

    if index == "_index_":
        del df["_index_"]

    return df_train, df_test