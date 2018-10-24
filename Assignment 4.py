# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:18:08 2018

@author: Ramesh
"""

import pandas as pd
data = pd.io.stata.read_stata('/Users/ramesh/Desktop/CDS.dta')
data.to_csv('/Users/ramesh/Desktop/CDS.csv')

a = pd.read_csv("/Users/ramesh/Desktop/CDS.csv")
b = pd.read_csv("/Users/ramesh/Desktop/CRSP.csv")
b = b.dropna(axis=1)
merged = a.merge(b, on='GVKEY')

i = 0

for d in b:
    quarter, year = d.split("\t")[1].split("Q")
    for j in range(3):
        month = (int(quarter) - 1) * 3 + j + 1
        print ("%s\t%s-01-%s" % (i, month, year))
        i += 1
        
        
df = pd.read_csv("/Users/ramesh/Desktop/CDS.csv")        
df.date = pd.to_datetime(df.date)
df['quarter'] = pd.PeriodIndex(df.date, freq='Q')

b = pd.read_csv("/Users/ramesh/Desktop/CRSP.csv")
b = b.dropna(axis=1)
merged = df.merge(b, on='date')
m = pd.concat
merged.to_csv("out.csv", index=False)

