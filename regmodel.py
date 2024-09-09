import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

### Reading data
iDir = 'input/'
pDir = 'plots/'

fnam = 'data.csv'
df = pd.read_csv(iDir + fnam,sep=',',header=0)
display(df.head())
display(df.tail())

### Feature inspection
fig = sns.pairplot(df,kind='reg', diag_kind='kde',corner=True,hue='group',aspect=1)
fig.savefig(pDir + 'pairplot-all.png',dpi=150)

key = df.columns[1:-1]

def plboxp(df,key,fnam):
    fig, ax = plt.subplots(1,len(key),figsize=(9,3))
    for i,k in enumerate(key):
        sns.boxplot(data=df,y=k,x='group',hue='group',fill=False,ax=ax[i],orient='v')

    fig.tight_layout()
    fig.savefig(pDir + fnam + '.png',dpi=150)

plboxp(df,key,'boxplot-all')

def outlim(ser):
    iqr = np.diff(np.percentile(ser,[25,75]))[0]
    med = np.median(ser)
    return(med-1.5*iqr,med+1.5*iqr)
dfn = pd.DataFrame()

for g in ['group1','group2']:
    kg = df.group == g
    dfi = df[kg]
    k = True
    for ke in key:
        lLim,uLim = outlim(dfi[ke])
        k1 = (dfi[ke]<=uLim) & (dfi[ke]>=lLim)
        k = k*k1

    dfn = pd.concat([dfn,dfi[k]])

print('   Size of original dataset:',df.shape[0])
print('Size of dataset wo outliers:',dfn.shape[0])
plboxp(dfn,key,'boxplot-clean')

fig = sns.pairplot(dfn,kind='reg', diag_kind='kde',corner=True,hue='group',aspect=1)
fig.savefig(pDir + 'pairplot-clean.png',dpi=150)


## Building regression model
npow =  {'group1':[3.0,1.0,-.5,2.0,2.0],
         'group2':[2.0,1.0,1.0,1.0,2.0]}

SEED = 1311131


y = dfn.dependent
X = pd.DataFrame()

model_name = ['LinearRegression','Ridge','ElasticNet']
model      = [LinearRegression(fit_intercept=True),
              Ridge(fit_intercept=True),
              ElasticNet(fit_intercept=True)]

p_dist = {'ElasticNet':{ 'alpha' : np.linspace(0.1,99.9,990), 'l1_ratio' : np.linspace(0.1,0.9,9),'tol' : [1e-1,1e-4]},
          'Ridge':{'alpha' : np.linspace(0.1,.9,990)},
         }

col = ['#1f77b4','#ff7f0e','#2ca02c']
mar = ['s','<','o']

fig,ax = plt.subplots(1,2,figsize=(9,4))
fiq,axq = plt.subplots(1,2,figsize=(9,4))

for i,g in enumerate(['group1','group2']):
    for ii,n in enumerate(npow[g]):
        X[key[ii]] = dfn[key[ii]]**n

    k = dfn.group == g

    X_train, X_test, y_train, y_test = train_test_split(X[k], y[k], test_size = 0.3 , random_state = SEED)

    vpr = []

    print('\n%8s metrics'%(g))
    s = [120,70,30]

    for ii,m in enumerate(model):

        if model_name[ii] != 'LinearRegression':
            rsh = RandomizedSearchCV(estimator=m, param_distributions=p_dist[model_name[ii]])
            rsh.fit(X_train, y_train)
            bp = rsh.best_params_
            m.set_params(**bp)
        m.fit(X_train,y_train)
        y_pred = m.predict(X_test)
        vpr.append(y_pred)
        form = '%%%is'%int(1/2*(len(model_name[ii])+8)+16)
        print('%12s'%model_name[ii])
        print('%16s:%8.5f'%('R^2',m.score(X_test,y_test)))
        print('%16s:%8.1f'%('RMSE',rmse(y_pred,y_test)))
        #ax[i].scatter(y_test,m.predict(X_test),ec=col[ii],s=s[ii],fc='white',alpha=1,label=model_name[ii])
        res = np.abs(y_test-m.predict(X_test))*10

        ###### Correlation Plots
        ax[i].scatter(y_test,y_pred,ec=col[ii],s=s[ii],fc=col[ii]+'22',label=model_name[ii])

        #### Q-Q Residuals Plots
        fiq = sm.qqplot(res, marker = mar[ii], markeredgecolor=col[ii],
                        markerfacecolor=col[ii]+'22', ax=axq[i], fit=True,
                        line='45', label=model_name[ii])
        axq[i].set_title(g)
        axq[i].legend()
        axq[i].set_aspect(1)
    print()
    ax[0].legend()
    xL = [min(y_test),max(y_test)]
    ax[i].plot(xL,xL,c='tab:red')
    ax[i].set_aspect(1)
    ax[i].set_title(g)

    dfr = pd.DataFrame(data=vpr).T
    dfr.columns = model_name
    dfr = pd.concat([y_test.reset_index(drop=True),dfr],axis=1)
    display(dfr.head())
    display(dfr.describe())



plt.show()
