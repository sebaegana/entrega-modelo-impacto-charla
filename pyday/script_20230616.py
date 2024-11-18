import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from pandas.api.types import CategoricalDtype
import numpy as np


data_01 = pd.read_excel('pyday/data/data_final.xlsx',sheet_name='data')  

data_01.dtypes

summary_data_01 = data_01.describe()

# Transformaciones

data_01["tipo_cliente"] = np.where(data_01["NPS"] >= 9, "Promotor", np.where((data_01["NPS"] < 9) & (data_01["NPS"] >= 7), "Neutro", "Detractor"))
data_01.groupby(by=["tipo_cliente"])["ID"].count()

cat_type_2 = CategoricalDtype(categories=["Detractor","Neutro","Promotor"], ordered=True)
data_01["tipo_cliente"] = data_01["tipo_cliente"].astype(cat_type_2)

cat_type = CategoricalDtype(categories=[0, 1, 2, 3, 4, 5,6,7,8,9,10], ordered=True)
data_01["NPS"] = data_01["NPS"].astype(cat_type)
data_01.groupby(by=["NPS"])["ID"].count()

data_01.groupby(by=["Market"])["ID"].count()
data_01 = pd.get_dummies(data_01, columns=["Market"], prefix=["d"],dtype=float, drop_first = False)


## Modelo 01

mod_log_01 = OrderedModel(data_01['NPS'],
                        data_01[['Driver 1', 'Driver 2', 'Driver 3']],
                        distr='logit')
                        
res_log_01 = mod_log_01.fit(method='bfgs', disp=False)
res_log_01.summary()

## Modelo 02

mod_log_02 = OrderedModel(data_01['tipo_cliente'],
                        data_01[['Driver 1', 'Driver 2', 'Driver 3']],
                        distr='logit')
                        
res_log_02 = mod_log_02.fit(method='bfgs', disp=False)
res_log_02.summary()

## Modelo 03

mod_log_03 = OrderedModel(data_01['tipo_cliente'],
                        data_01[['Driver 1', 'Driver 2', 'Driver 3', 'Edad', "d_MEX"]],
                        distr='logit')
                        
res_log_03 = mod_log_03.fit(method='bfgs', disp=False)
res_log_03.summary()

# Modelo por ubicación

data_us = data_01[data_01["d_US"] == 1]

mod_log_04 = OrderedModel(data_us['tipo_cliente'],
                        data_us[['Driver 1', 'Driver 2', 'Driver 3', 'Edad']],
                        distr='logit')
                        
res_log_04 = mod_log_04.fit(method='bfgs', disp=False)
res_log_04.summary()

### ¿Cómo calcular la probabilidad?

data_reg_04 = mod_log_04.fit(method='bfgs', disp=False).params

1. / (1. + np.exp(-0.003110))
1. / (1. + np.exp(data_reg_04[3]))

