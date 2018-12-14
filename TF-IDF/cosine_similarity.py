"""
Goal: this code is for buidling cosine similarity function and build the efficient comparison matrix
Author: Shenghan Zhang
coding:utf-8
date: 4th, Dec, 2018
"""

import numpy as np
import pandas as pd

class CompareSimilarity():
    def __init__(self,list_id,list_content):
        # here the order of list_id must be the same as list_content comparing with the original dataframe
        self.list_id=list_id
        self.list_content=list_content
        # instuct a new dataframe to store the outcome of comparison
        self.matrix_df=pd.DataFrame(columns=list_id,index=list_id)

    def getCosine(self,vec_q, vec_d):
        # this is the function of getting cosine similarity
        sum_num = 0
        sum_denom_q = 0
        sum_denom_d = 0
        for i in range(len(vec_q)):
            sum_num = vec_q[i] * vec_d[i] + sum_num
            sum_denom_q = np.square(vec_q[i]) + sum_denom_q
            sum_denom_d = np.square(vec_d[i]) + sum_denom_d

        sum_denom = (np.sqrt(sum_denom_q)) * (np.sqrt(sum_denom_d))
        cos_q_d = sum_num / sum_denom
        return cos_q_d

    def cos_matrix(self):
        # the function to get the matrix of cosine similarity comaprison outcome of different pair
        id_len=len(self.list_id)
        #print(id_len)
        for no in range(id_len):
            #print("no=%d"%no)
            i=1
            while i<(id_len-no):
                #print("no+i=%d"%(no+i))
                CosSim = self.getCosine(self.list_content[no],self.list_content[no+i])
                column_no=self.list_id[no+i]
                index_no=self.list_id[no]
                self.matrix_df[column_no][index_no]=CosSim
                i=i+1

        return self.matrix_df