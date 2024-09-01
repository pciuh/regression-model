#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:25:41 2021

@author: pciuh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.metrics import mean_absolute_error

iDir = 'input/'
pDir = 'plots/'

TST = True

fnam = 'data.csv'
df = pd.read_csv(iDir + fnam,sep=',',header=0)

grn = ['group1','group2']

IND = 0
SEED = [306,921]
TSIZ = [0.33,0.25]
rLim = [30,15]

ofnam = grn[IND]+'-report.txt'

dfa = df[df.group==grn[IND]].drop(['group'],axis=1)
print(dfa.head())

df_c = dfa.corr(method='spearman')

print(grn[IND])
print(dfa.isnull().sum())

Y = dfa.dependent.reset_index(drop=True)

vheads = np.array([['time','var1','var2','var3','var4'],
                   ['time','var1','var2','var3','var4']],dtype=object)
vnpow =  np.array([[3.0,1.0,-.5,2.5,2.0],
                   [3.0,1.0,1.0,2.5,2.0]],dtype=object)

#print(vheads[IND])
heads = vheads[IND]
#heads = ['time','var1','var2','var3','var4']

npow  = vnpow[IND]

nvar = len(heads)
data = np.zeros((len(dfa['time']),nvar))
for i,h in enumerate(heads):
    data[:,i] = dfa[h]**npow[i]

dc = dict(zip(heads,data.T))
X = pd.DataFrame(dc)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = TSIZ[IND] , random_state = SEED[IND]) 


of = open(ofnam,'w')

of.write('Training Data Count: {}\n'.format(X_train.shape[0]))
of.write('Testing Data Count: {}\n'.format(X_test.shape[0]))

X_train = sm.add_constant(X_train)
results = sm.OLS(y_train, X_train).fit()

of.write(results.summary().as_text())

X_test = sm.add_constant(X_test)
Y_preds = results.predict(X_test)

mSiz = 100*np.abs(y_test-Y_preds)/max(y_test)
#mSiz = (y_test-Y_preds)

fig,ax = plt.subplots(1,figsize=(5,5))
#plt.style.use('seaborn')
scp = ax.scatter(y_test,Y_preds,c=mSiz,cmap='viridis',alpha=1.0)
cba = plt.colorbar(scp,shrink=.75,label='Residuals / Predict Max [%]')
scp.set_clim(0,5.0)
ax.set_title(grn[IND])
ax.plot(y_test,y_test,'-r')
ax.set_xlabel('Test')
ax.set_ylabel('Prediction')
ax.set_aspect(1)
ax.grid()

fig.savefig(grn[IND]+'-reg.png',dpi=300,bbox_inches='tight')

bns = np.linspace(-rLim[IND],rLim[IND],13)

fig,ax = plt.subplots(1,figsize=(5,5))
num,bns,_ = ax.hist(y_test - Y_preds,bns,rwidth=.96,color='#6688aa',align='mid')
ax.set_title(grn[IND])
ax.set_xlabel('Residuals')
ax.set_ylabel('Counts')
ax.grid(axis='y', alpha=0.75)
ax.set_xticks(bns[::2])


coef=results.params
of.write('\n\nErrors:\n')
of.write(' MAE: {:6.2f}\n'.format(mean_absolute_error(y_test, Y_preds)))
of.write(' MSE: {:6.2f}\n'.format(mse(y_test, Y_preds)))
of.write('RMSE: {:6.2f}\n'.format(rmse(y_test, Y_preds)))
of.write('\n')

fig.savefig(grn[IND]+'-residuals.png',dpi=300,bbox_inches='tight')

for i,c in enumerate(coef):
    if i<1:
        of.write('dependent = %.3f'%c)
    else:
        if c>0:
            of.write(' + ')

        of.write('%.3f x %s ^ %.1f'%(c,heads[i-1],npow[i-1]))

of.close()

if TST:
    #fig1,ax = plt.subplots(1,figsize=(9,9))
    #sns.heatmap(df_c,vmin=-1,vmax=1,cmap='viridis',annot=True,linewidth=.1,ax=ax, cbar_kws={'shrink':.7})
#
    #ax.set_aspect(1)
    #fig1.savefig('heatmap-'+grn[IND]+'.png',dpi=300)
#
#    fig,ax = plt.subplots(1,figsize=(6,6))
#    fig=sns.pairplot(df,kind='reg', diag_kind='kde',hue='group')
#    fig.savefig(pDir + 'pairplot.png',dpi=150)

    vec = df.columns[1:-1]
    print(vec)
    fig, ax = plt.subplots(1,len(vec),figsize=(12,4))
    for i,v in enumerate(vec):
        sns.boxplot(data=df,y=v,x='group',hue='group',fill=False,ax=ax[i],orient='v')
    fig.tight_layout()
    fig.savefig(pDir + 'boxplot.png',dpi=150)


def outlim(ser):
    iqr = np.diff(np.percentile(ser,[25,75]))[0]
    med = np.median(ser)
    return(med-1.5*iqr,med+1.5*iqr)


#vec = ['var1','var3']
dfn = pd.DataFrame()
for g in ['group1','group2']:
    kg = df.group == g
    dfi = df[kg]
    k = True
    for ke in vec:
        lLim,uLim = outlim(dfi[ke])
        k1 = (dfi[ke]<=uLim) & (dfi[ke]>=lLim)
        k = k*k1

    dfn = pd.concat([dfn,dfi[k]])

print(df.shape[0],dfn.shape[0])

fig, ax = plt.subplots(1,len(vec),figsize=(12,4))
for i,v in enumerate(vec):

    sns.boxplot(data=dfn,y=v,x='group',hue='group',fill=False,ax=ax[i],orient='v')
    fig.tight_layout()
    fig.savefig(pDir + 'boxplot-no.png',dpi=150)
#fig,ax = plt.subplots(2)
#sns.boxplot(data=df[k],y=df[k][key],x='group',hue='group',fill=False,ax=ax,orient='v')
#plt.show()
