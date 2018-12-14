"""
Goal: the function of getting the result of tf-idf comparison and the evaluation of the result
Author: Shenghan Zhang
coding:utf-8
date: 5th, Dec, 2018
"""

import pandas as pd

class ResultProcessor():
    def __init__(self,df_sim_matrix,df_gs):
        self.df_sim_matrix=df_sim_matrix
        self.df_gs=df_gs
        self.list_id = list(self.df_sim_matrix.columns.values)

    def convert_gold_standard(self):
        # get the pair of each comparison one by one with cosine similarity
        pair_df = pd.DataFrame(columns=['Comparison_1','Comparison_2','cosine_similarity'])
        list_1=self.df_gs['Comparison_1']
        list_2=self.df_gs['Comparison_2']
        list_sim=list()
        length=len(list_1)
        for n in range(length):
            if self.df_sim_matrix[list_1[n]][list_2[n]]>=0:
                list_sim.append(self.df_sim_matrix[list_1[n]][list_2[n]])
            else:
                list_sim.append(self.df_sim_matrix[list_2[n]][list_1[n]])
        pair_df['Comparison_1']=list_1
        pair_df['Comparison_2']=list_2
        pair_df['cosine_similarity']=list_sim
        return pair_df

    def set_threshold(self,threshold,pair_df):
        pair_df=pair_df.rename(columns={"cosine_similarity":"Label_predict"})
        list_sim=pair_df["Label_predict"]
        list_predict=list()
        for n in range(len(list_sim)):
            if list_sim[n] >=threshold:
                list_predict.append(True)
            else:
                list_predict.append(False)

        pair_df["Label_predict"]=list_predict
        return pair_df

    def get_best_threshold(self,start,step,pair_df):
        threshold=start
        base=0
        while threshold < 1:
            df_predict=self.set_threshold(threshold,pair_df)
            eo=EvalOutcome(df_predict,self.df_gs)
            f_measure=eo.eval_f_measure()
            if f_measure < base:
                threshold=threshold-step
                break
            else:
                base=f_measure
            threshold=threshold+step
        return threshold


class EvalOutcome():
    def __init__(self, df_predict, df_gs):
        self.df_predict = df_predict
        self.df_gs = df_gs
        self.index_size = df_gs.index.size

    def get_no(self):
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for no in range(self.index_size):
            gold_standard = self.df_gs['Label'].iloc[no]
            # print("gold_standard:"+str(gold_standard))
            prediction = self.df_predict['Label_predict'].iloc[no]
            # print("prediction:"+str(prediction))
            if (gold_standard == prediction) & (gold_standard == True):
                TP = TP + 1
                # print("TP:%d"%TP)
            elif (gold_standard == prediction) & (gold_standard == False):
                TN = TN + 1
                # print("TN:%d"%TN)
            elif (gold_standard != prediction) & (gold_standard == True):
                FN = FN + 1
                # print("FN:%d"%FN)
            else:
                FP = FP + 1
                # print("FP:%d"%FP)
        # print("TP: %d"% TP)


        return TP, TN, FN, FP

    def eval_precision(self):
        TP, TN, FN, FP = self.get_no()
        print("TP= %d, TN= %d, FN= %d, FP= %d" % (TP, TN, FN, FP))
        precision = TP / (TP + FP)
        return precision

    def eval_recall(self):
        TP, TN, FN, FP = self.get_no()
        print("TP= %d, TN= %d, FN= %d, FP= %d" % (TP, TN, FN, FP))
        recall = TP / (TP + FN)
        return recall

    def eval_f_measure(self):
        precision = self.eval_precision()
        recall = self.eval_recall()
        f_measure = 2 * (precision * recall) / (precision + recall)
        return f_measure