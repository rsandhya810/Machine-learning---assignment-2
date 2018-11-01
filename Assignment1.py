import pandas as pd
import csv
import operator
import tabulate

#To read the csv file
df = pd.read_csv("/Users/ramesh/Desktop/secdata.csv")
print (df)


#To show the summary statistics for each variable
df_ss = df.describe(include='all')
df_ss

# To transppose the dataframe
df_t = df.describe().transpose()
df_t

# to find the median
df_ss1 = df.median()
df_ss1

#To find the count of missing/Nan values
df_ss2 = df.isnull().sum()
df_ss2.shape

#To drop the column with more than 50% missings
df_2 = df_ss.loc[:, df.isnull().sum() < 0.5*df.shape[0]]
df_2


# To convert and save the result(df) to csv file
df_2.to_csv('data.csv')


#To convert it to table

with open('/Users/ramesh/Desktop/data.csv') as inf:
    reader = sorted(csv.reader(inf), key=operator.itemgetter(1))
    headers = next(reader)
    print(tabulate([(row[0], row[1], row[2]-row[3],
                     "{:>2} : {:2}".format(row[2], row[3])) for row in reader],
                   headers=headers))
    
df.describe(include='all').style

def color(val):
    color = 'red' if val < 0 else 'green'
    return 'color: %s' % color
table = df.describe(include='all').style.applymap(color)
table
