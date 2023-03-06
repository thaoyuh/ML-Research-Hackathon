#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:09:25 2023

@author: longjiaoli
"""

####################
### Imports
####################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plotly.offline import iplot
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.formula.api as sm
import datetime
#import sklearn as skl
from pandarallel import pandarallel

pandarallel.initialize()

####################
### Read data
####################
df = pd.read_csv('wildfire_data.csv')
print('Number of fires:', df.shape[0])
df.head()

####################
### Add climate features
####################

# load the data
# ref: https://www.ncei.noaa.gov/pub/data/cirs/climdiv/state-readme.txt
names = ['code','jan','feb','mar','apr','may','jun',
         'jul','aug','sep','oct','nov','dec']
types = {'code':str,'jan':float,'feb':float,'mar':float,
        'apr':float,'may':float,'jun':float,
         'jul':float,'aug':float,'sep':float,'oct':float,
        'nov':float,'dec':float}
df_tmp = pd.read_csv("climdiv-tmpcst-v1.0.0-20230106.txt", sep = "  ", 
                     names = names, header = None, 
                     dtype=types, engine='python')
df_pcp = pd.read_csv("climdiv-pcpnst-v1.0.0-20230106.txt", sep = "  ", 
                     names = names, header = None, 
                     dtype=types, engine='python')
df_pdsi = pd.read_csv("climdiv-pdsist-v1.0.0-20230106.txt", sep = "  ", 
                      names = names, header = None, 
                      dtype = types, engine='python')

# compute the yearly avaerage
df_tmp['ave_year'] = df_tmp.iloc[:,1:12].mean(axis = 1)
df_pcp['ave_year'] = df_pcp.iloc[:,1:12].mean(axis = 1)
df_pdsi['ave_year'] = df_pdsi.iloc[:,1:12].mean(axis = 1)

# extract the year from nama code
df_tmp['year'] = df_tmp['code'].map(lambda x: int(x[-4:]))
df_pcp['year'] = df_pcp['code'].map(lambda x: int(x[-4:]))
df_pdsi['year'] = df_pdsi['code'].map(lambda x: int(x[-4:]))

# only keep the data from 1992 to 2015
df_tmp.query('1991 < year < 2016',inplace=True)
df_pcp.query('1991 < year < 2016',inplace=True)
df_pdsi.query('1991 < year < 2016',inplace=True)

# match the code with the state name in the wildfire data
state_code = {'001':'AL','002':'AZ','003':'AR','004':'CA','005':'CO',
'006':'CT','007':'DE','008':'FL','009':'GA','010':'ID','011':'IL','012':'IN',
'013':'IA','014':'KS','015':'KY','016':'LA','017':'ME','018':'MD',
'019':'MA','020':'MI','021':'MN','022':'MS','023':'MO','024':'MT',
'025':'NE','026':'NV','027':'NH','028':'NJ','029':'NM','030':'NY',
'031':'NC','032':'ND','033':'OH','034':'OK','035':'OR','036':'PA',
'037':'RI','038':'SC','039':'SD','040':'TN','041':'TX','042':'UT',
'043':'VT','044':'VA','045':'WA','046':'WV','047':'WI','048':'WY',
'050':'AK'
}
df_tmp['state'] = df_tmp['code'].map(lambda x: state_code[x[0:3]] if x[0:3] in state_code.keys() else 'other')
df_pcp['state'] = df_pcp['code'].map(lambda x: state_code[x[0:3]] if x[0:3] in state_code.keys() else 'other')
df_pdsi['state'] = df_pdsi['code'].map(lambda x: state_code[x[0:3]] if x[0:3] in state_code.keys() else 'other')

# compute yearly averages
df['tmp_yearly'] = 0.00
df['pcp_yearly'] = 0.00
df['pdsi_yearly'] = 0.00

def match_tmp(x):
    if x['STATE'] not in ['PR','HI','DC']:
        tmp = df_tmp['ave_year'][(df_tmp['state'] == x['STATE']) & (df_tmp['year'] == x['FIRE_YEAR'])].iloc[0]
    else:
        tmp = -99.99
    return tmp

def match_pcp(x):
    if x['STATE'] not in ['PR','HI','DC']:
        pcp = df_pcp['ave_year'][(df_pcp['state'] == x['STATE']) & (df_pcp['year'] == x['FIRE_YEAR'])].iloc[0]
    else:
        pcp = -99.99
    return pcp

def match_pdsi(x):
    if x['STATE'] not in ['PR','HI','DC']:
        pdsi = df_pdsi['ave_year'][(df_pdsi['state'] == x['STATE']) & (df_pdsi['year'] == x['FIRE_YEAR'])].iloc[0]
    else:
        pdsi = -99.99
    return pdsi

df['tmp_yearly'] = df.loc[:,['STATE','FIRE_YEAR']].parallel_apply(match_tmp, axis = 1)
df['pcp_yearly'] = df.loc[:,['STATE','FIRE_YEAR']].parallel_apply(match_pcp, axis = 1)
df['pdsi_yearly'] = df.loc[:,['STATE','FIRE_YEAR']].parallel_apply(match_pdsi, axis = 1)

# compute monthly average
df['DATE'] = pd.to_datetime(df['DISCOVERY_DATE'], unit='D', origin='julian')
df['MONTH'] = df['DATE'].dt.month

df['tmp_monthly'] = 0.00
df['pcp_monthly'] = 0.00
df['pdsi_monthly'] = 0.00


def match_tmp_month(x):
    if x['STATE'] not in ['PR','HI','DC']:
        tmp = df_tmp[(df_tmp['state'] == x['STATE']) & (df_tmp['year'] == x['FIRE_YEAR'])].iloc[0,x['MONTH']]
    else:
        tmp = -99.99
    return tmp

def match_pcp_month(x):
    if x['STATE'] not in ['PR','HI','DC']:
        pcp = df_pcp[(df_pcp['state'] == x['STATE']) & (df_pcp['year'] == x['FIRE_YEAR'])].iloc[0,x['MONTH']]
    else:
        pcp = -99.99
    return pcp

def match_pdsi_month(x):
    if x['STATE'] not in ['PR','HI','DC']:
        pdsi = df_pdsi[(df_pdsi['state'] == x['STATE']) & (df_pdsi['year'] == x['FIRE_YEAR'])].iloc[0,x['MONTH']]
    else:
        pdsi = -99.99
    return pdsi

df['tmp_monthly'] = df.loc[:,['STATE','FIRE_YEAR','MONTH']].parallel_apply(match_tmp_month, axis = 1)
df['pcp_monthly'] = df.loc[:,['STATE','FIRE_YEAR','MONTH']].parallel_apply(match_pcp_month, axis = 1)
df['pdsi_monthly'] = df.loc[:,['STATE','FIRE_YEAR','MONTH']].parallel_apply(match_pdsi_month, axis = 1)



####################
### Add neiborhood containment time within a year
####################

# load the data with climate features
df = pd.read_csv('wildfire_data_with_climate.csv')
df.DATE = pd.to_datetime(df.DATE,format = "%Y-%m-%d")

# compute the containment time
lDAY_TO_CONT=[]
lHOUR_TO_CONT = []
for i in df.index:
    vDay2cont = df.loc[i,'CONT_DATE']-df.loc[i,'DISCOVERY_DATE']
    vHour2cont = (24 - df.loc[i,'DISCOVERY_TIME']) + (vDay2cont-1)*24 + df.loc[i,'CONT_TIME']
    lDAY_TO_CONT.append(round(vDay2cont))
    lHOUR_TO_CONT.append(round(vHour2cont)) # this also rounds it to full hours, may be relaxed
df['DAY_TO_CONT'] = lDAY_TO_CONT
df['HOUR_TO_CONT'] = lHOUR_TO_CONT

# compute the average neiborhood containment time within the one year before the current fire
def nearby_contain_hour(x):
    time_thre = datetime.timedelta(days = 183) # set half an year as threshold
    index = (df.LATITUDE - x['LATITUDE'] != 0) & (abs(df.LATITUDE - x['LATITUDE']) < 0.5) & (abs(df.LONGITUDE - x['LONGITUDE']) < 0.5) & (abs(df.DATE - x['DATE']) < time_thre)
    if df[index].empty:
        return None
    else:
        return df[index].HOUR_TO_CONT.mean()


df['NEARBY_HOUR_TO_CONT'] = 0
df['NEARBY_DAY_TO_CONT'] = 0


df['NEARBY_HOUR_TO_CONT'] = df.loc[:,['LATITUDE','LONGITUDE','DATE']].parallel_apply(nearby_contain_hour, axis = 1)
df['NEARBY_DAY_TO_CONT'] = round(df.NEARBY_HOUR_TO_CONT/24)

    
# write to csv
df.to_csv('wildfire_data_climate_nearby.csv',index = False)
