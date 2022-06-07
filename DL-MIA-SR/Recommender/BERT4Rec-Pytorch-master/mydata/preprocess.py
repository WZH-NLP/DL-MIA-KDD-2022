# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import datetime


def preprocess(dataset, member, nonmember, nonmem_train):
    data = pd.read_csv(dataset, sep=',', header=None, skiprows=1)
    data.columns = ['SessionID', 'ItemID', 'Rating', 'Time']
    # split member and nonmember 5-5
    middle = round(data.SessionID.nunique() / 2)
    member_data = pd.DataFrame()
    nonmember_data = pd.DataFrame()

    data = data.groupby('SessionID')
    for index, value in data:
        if index < middle:
            member_data = member_data.append(value)
        else:
            nonmember_data = nonmember_data.append(value)

    n_data = nonmember_data.sort_values(by=['SessionID', 'Time'], ascending=True)
    n_data = n_data.groupby('SessionID')
    n_train = pd.DataFrame()
    for index, value in n_data:
        n_train = n_train.append(value.iloc[:-2, :])
    n_train.to_csv(nonmem_train, sep=',', index=False)

    # Convert To CSV
    print('Member has', len(member_data), 'Events, ', member_data.SessionID.nunique(), 'Sequences, and',
          member_data.ItemID.nunique(), 'Items\n\n')
    member_data.to_csv(member, sep=',', index=False)
    print('Nonmember has', len(nonmember_data), 'Events, ', nonmember_data.SessionID.nunique(), 'Sequences, and',
          nonmember_data.ItemID.nunique(), 'Items\n\n')
    nonmember_data.to_csv(nonmember, sep=',', index=False)


if __name__ == '__main__':
    # # m: ml-1m a: Amazon
    # dataBefore_root = "../../../Dataset/data/processed_movielens/"
    # dataAfter_root = "processed/"
    # m_shadow_dataset = os.path.join(dataBefore_root, "ml-1m_shadow")
    # m_shadow_member = os.path.join(dataAfter_root, "ml-1m_Smember")
    # m_shadow_nonmember = os.path.join(dataAfter_root, "ml-1m_Snonmem")
    # m_shadow_nonmem_train = os.path.join(dataAfter_root, "ml-1m_Snonmem_train")
    # preprocess(m_shadow_dataset, m_shadow_member, m_shadow_nonmember, m_shadow_nonmem_train)
    #
    # m_target_dataset = os.path.join(dataBefore_root, "ml-1m_target")
    # m_target_member = os.path.join(dataAfter_root, "ml-1m_Tmember")
    # m_target_nonmember = os.path.join(dataAfter_root, "ml-1m_Tnonmem")
    # m_target_nonmem_train = os.path.join(dataAfter_root, "ml-1m_Tnonmem_train")
    # preprocess(m_target_dataset, m_target_member, m_target_nonmember, m_target_nonmem_train)

    # m: ml-1m a: Amazon
    dataBefore_root = "../../../Dataset/data/processed_amazon/"
    dataAfter_root = "processed/"
    m_shadow_dataset = os.path.join(dataBefore_root, "beauty_shadow")
    m_shadow_member = os.path.join(dataAfter_root, "beauty_Smember")
    m_shadow_nonmember = os.path.join(dataAfter_root, "beauty_Snonmem")
    m_shadow_nonmem_train = os.path.join(dataAfter_root, "beauty_Snonmem_train")
    preprocess(m_shadow_dataset, m_shadow_member, m_shadow_nonmember, m_shadow_nonmem_train)

    m_target_dataset = os.path.join(dataBefore_root, "beauty_target")
    m_target_member = os.path.join(dataAfter_root, "beauty_Tmember")
    m_target_nonmember = os.path.join(dataAfter_root, "beauty_Tnonmem")
    m_target_nonmem_train = os.path.join(dataAfter_root, "beauty_Tnonmem_train")
    preprocess(m_target_dataset, m_target_member, m_target_nonmember, m_target_nonmem_train)
