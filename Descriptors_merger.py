#!/usr/bin/env python
# coding: utf-8

import pandas as pd

file0="VII_Descriptors_monomer.csv"
file1="VII_Descriptors_MD.csv"
file2="VII_Descriptors_Mordred.csv"

bdt0=pd.read_csv(path+file0)
bdt1=pd.read_csv(path+file1)
bdt2=pd.read_csv(path+file2)

bdt11=bdt1.drop(df1.columns[0:2], axis=1)
bdt21=bdt2.drop(df2.columns[0:3], axis=1)
bdt_des=bdt0.join(df11)
bdt_des=bdt_des.join(df21)
bdt_des.to_csv("./dataset/Descriptors/VII_Descriptors.csv",index=None)