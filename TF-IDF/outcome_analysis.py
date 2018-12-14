"""
Goal: the function of getting the result of tf-idf comparison and the evaluation of the result
Author: Shenghan Zhang
coding:utf-8
date: 5th, Dec, 2018
"""

import pandas as pd

class OverallOutcomeProcessor():
    def __init__(self,matrix_df):
        self.matrix_df=matrix_df
        self.list_id = list(self.matrix_df.columns.values)
        self.length=len(self.list_id)

    def get_result_df(self, threshold):
        for no in range(self.length):
            i = 1
            while i <= (self.length - no):
                column_no = self.list_id[no + i]
                index_no = self.list_id[no]
                if self.matrix_df[column_no][index_no] >= threshold:
                    self.matrix_df[index_no][column_no] = True
                else:
                    self.matrix_df[index_no][column_no] = False

        return self.matrix_df

    def convert_pair(self,new_matrix_df):
        # get the pair of each comparison one by one,not by matrix
        pair_df=pd.DataFrame(columns=['Comparison_1','Comparison_2','Label'])
        index_all=0
        for no in range(self.length):
            i=1
            while i <=(self.length - no):
                index_no = self.list_id[no + i]
                column_no = self.list_id[no]
                data=[[column_no,index_no,new_matrix_df[column_no][index_no]]]
                columns=['Comparison_1','Comparison_2','Label']
                index_all=index_all+1
                one_pair_df=pd.DataFrame(columns=columns,index=index_all,data=data)
                frames=[pair_df,one_pair_df]
                pair_df=pd.concat(frames)

        return pair_df

    def convert_pair_cosine(self,matrix_df):
        # get the pair of each comparison one by one with cosine similarity
        pair_df = pd.DataFrame(columns=['Comparison_1','Comparison_2','cosine_similarity'])
        index_all = 0
        for no in range(self.length):
            i = 1
            while i <= (self.length - no):
                index_no = self.list_id[no]
                column_no = self.list_id[no+i]
                data = [[column_no, index_no, matrix_df[column_no][index_no]]]
                columns = ['Comparison_1','Comparison_2', 'cosine_similarity']
                index_all = index_all + 1
                one_pair_df = pd.DataFrame(columns=columns, index=index_all, data=data)
                frames = [pair_df, one_pair_df]
                pair_df = pd.concat(frames)

        return pair_df


class EvalOutcome():
    def __init__(self,matrix_df,df_gold):
        self.matrix_df=matrix_df
        self.df_gold=df_gold
        self.index_size=df_gold.index.size
        data=[]
        for no in range(self.index_size):
            product_1=df_gold['Comparison_1'].iloc[no]
            product_2=df_gold['Comparison_2'].iloc[no]
            if matrix_df[product_1][product_2] == True or matrix_df[product_1][product_2] == False:
                list_predict=[product_1,product_2,matrix_df[product_1][product_2]]
            else:
                list_predict = [product_1, product_2, matrix_df[product_2][product_1]]
            data.append(list_predict)
        columns=['Comparison_1','Comparison_2','Label_predict']
        self.df_predict=pd.DataFrame(columns=columns,data=data)

    def get_no(self):
        TP=0
        TN=0
        FN=0
        FP=0
        for no in range(self.index_size):
            gold_standard=self.df_gold['Label'].iloc[no]
            prediction=self.df_predict['Label_predict'].iloc[no]
            if gold_standard == prediction & gold_standard==True:
                TP=TP+1
            elif gold_standard == prediction & gold_standard==False:
                TN=TN+1
            elif gold_standard !=prediction & gold_standard==True:
                FN=FN+1
            elif gold_standard !=prediction & gold_standard==False:
                FP=FP+1

        return TP,TN,FN,FP

    def eval_precision(self):
        TP,TN,FN,FP=self.get_no()
        precision=TP/(TP+FP)
        return precision

    def eval_recall(self):
        TP, TN, FN, FP = self.get_no()
        recall = TP / (TP + FN)
        return recall

    def eval_f_measure(self):
        precision=self.eval_precision()
        recall=self.eval_recall()
        f_measure=2*(precision*recall)/(precision+recall)
        return f_measure



