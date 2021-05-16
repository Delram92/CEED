# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 18:44:36 2021

@author: Yohana Delgado Ramos -Milena Beltran

Proyecto final Machine learning 
Metodologia CRISP-DM
"""

"""Revision y exploracion de los datos """
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
import calendar
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
#from keras.models import Sequential
#from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler
# Preprocesado y modelado
# ==============================================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,confusion_matrix,precision_recall_curve,precision_score, accuracy_score, f1_score,  recall_score , r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.inspection import permutation_importance

import multiprocessing

warnings.filterwarnings('ignore')

#
PASOS=7
def strip_obj(col):
    #print(col.dtype)
    if col.dtype==object :
        col=(col.astype(str)
                   .str.strip()
                   .replace({'': np.nan}))
    if col.dtype!=object:
        col=(col.astype(str)
                   .str.strip()
                   .replace({'': np.nan}))
        col=col.astype(np.int64)
    
    return col

 
def validate_data():
    dataset = pd.read_csv('./dataset/Estructura_CEED.csv',";", thousands = ' ')
    dataset = dataset[dataset.DPTO_MPIO==11001]
    dataset = dataset[dataset.ANO_CENSO>=2016]


    #print(dataset.columns)
    dataset=dataset.apply(strip_obj, axis=0)
    
  #  print(dataset.values.str.strip())
    #print(dataset.head())
    #print(dataset.info())
    #print(1)
    
   # print(dataset['PRECIOVTAX'].describe())
    sns.distplot(dataset['PRECIOVTAX'])
    """Analisis multivariable
        Correlacion entre variables"""
    #print(dataset.info)
    corr_per = dataset.corr()
    
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr_per)
    plt.show()
    
    corr_sper = dataset.corr(method='spearman')
    
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr_sper)
    plt.show()
    return dataset

def new_dataset(dataset):
    
    atributos = ['PRECIOVTAX','REGION','ANO_CENSO','OB_FORMAL','AMPLIACION','ESTADO_ACT','ESTRATO','USO','ANOINICIO','AREATOTZC','UNIDADESGA',
'PRECIOUNIG','TIPOVALOR','SIS_CONSTR','MANO_OBRAP','MANO_OBRAT','AREA_LOTE','AREAVENDIB','NUMUNIDEST','NUMUNIVEN',
'NUMUNIXVEN','NRO_EDIFIC','NRO_PISOS','TIPOVRDEST','AREATOTCO2','AREAVENUNI','TIPOVIVI','RANVIVI','Destino2','LOCALIDAD2','FECHADILI','MESINICIO']
    dataset2=  dataset[atributos]
    
    df = dataset2.copy()
    monthlist = dataset2['MESINICIO'].astype(str)
    mlist = []
    
    m = np.array(monthlist)
    for mi in m:
        if(len(mi)<2 and mi!='0'):
            mi='0'+mi
        if(mi==0):
            mi=(mi.astype(str)
                    .str.strip()
                    .replace({'0': np.nan})).astype(np.int64)
        mlist.append(mi)
    df['MESINICIO'] =  mlist
    dataset2=df


    dataset2['FECHAINICIO']= (dataset2['ANOINICIO'].astype(str)+(dataset2['MESINICIO']).astype(str)).astype(np.int64)
    print(str(dataset2['FECHAINICIO']))
    
    dataset2['lenfecha']=dataset2['FECHAINICIO'].astype(str).map(len);
    
    #int(str(dataset_2['FECHAINICIO']))
    dataset2['FECHADILI'] = pd.to_datetime(dataset2.FECHADILI)
   # dataset2.set_index('FECHADILI', inplace=True)
 
  
    dataset2['y']=(dataset2['PRECIOVTAX'].astype(np.int64))
    
    dataset2 = dataset2.drop(dataset2[dataset2['lenfecha']<6].index)
    #dataset2 = dataset2[dataset2.DPTO_MPIO==11001]
    #dataset2=dataset2.drop(['ANOINICIO', 'MESINICIO','lenfecha'], axis=1)    


 
 
    return dataset2

def Corre_data(dataset):
    corr_per_2 = dataset.corr()

    
    plt.figure(figsize=(12, 9))
    plt.title("Correlación de pearson -Dataset nuevo")
    sns.heatmap(corr_per_2)
    plt.show()
    
    corr_sper_2 = dataset.corr(method='spearman')
    
    plt.figure(figsize=(12, 9))
    plt.title("Correlación de spearman -Dataset nuevo")

    sns.heatmap(corr_sper_2)
    plt.show()
def model_1(dataset): ##DESCARTADO, DEMORADO Y MALAS ESTADISTICASA
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import LinearRegression
   # dataset=dataset.drop(['FECHADILI'], axis=1)    
        #normalizar datos
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #dataset = scaler.fit_transform(dataset)
    atributos = ['OB_FORMAL','AMPLIACION','ESTADO_ACT','ESTRATO','USO','ANOINICIO','AREATOTZC','UNIDADESGA',
'PRECIOUNIG','TIPOVALOR','SIS_CONSTR','MANO_OBRAP','MANO_OBRAT','AREA_LOTE','AREAVENDIB','NUMUNIDEST','NUMUNIVEN',
'NUMUNIXVEN','NRO_EDIFIC','NRO_PISOS','TIPOVRDEST','AREATOTCO2','AREAVENUNI','TIPOVIVI','RANVIVI','Destino2']
    
    print("Hola ssdeeds")
    
   # logreg = LinearRegression(solver='lbfgs',random_state = 0)
    reg = LinearRegression()
    
    
    x = dataset[atributos]
    y = dataset['PRECIOVTAX']
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
    
    reg.fit(X_train, y_train)
    
    print("Metricas con regresion lineal\n")
    
    print('Train: ', reg.score(X_train,y_train))
    print('Test: ', reg.score(X_test,y_test))
    

    y_pred = reg.predict(X_test)
   # print(y_pred)
    
    df_predicciones = pd.DataFrame({'precio' : y_test, 'prediccion' : y_pred})
    print( df_predicciones.head())


    # r = np.random.randint(0, 21613)
   
    # x_test = (dataset[r:r+1].drop(columns=['PRECIOVTAX']))
    # print("Predict price: ", logreg.predict(x_test))
    # print("Real price: ", logreg.at[r,'PRECIOVTAX'])

def model_2(dataset):
   from sklearn.tree import DecisionTreeRegressor
   
   atributos = ['OB_FORMAL','AMPLIACION','ESTADO_ACT','ESTRATO','USO','FECHAINICIO','AREATOTZC','UNIDADESGA',
'PRECIOUNIG','TIPOVALOR','SIS_CONSTR','MANO_OBRAP','MANO_OBRAT','AREA_LOTE','AREAVENDIB','NUMUNIDEST','NUMUNIVEN',
'NUMUNIXVEN','NRO_EDIFIC','NRO_PISOS','TIPOVRDEST','AREATOTCO2','AREAVENUNI','TIPOVIVI','RANVIVI','Destino2']
    
   x = dataset[atributos]
   y = dataset['PRECIOVTAX']
    
   # logreg = LinearRegression(solver='lbfgs',random_state = 0)
   reg = DecisionTreeRegressor()
   X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
    
   reg.fit(X_train, y_train)
     
   print("Metricas con arboles de decision\n")

    
   print('Train: ', reg.score(X_train,y_train))
   print('Test: ', reg.score(X_test,y_test))

   y_pred = reg.predict(X_test)
   print(y_pred)
   
   df_predicciones = pd.DataFrame({'precio' : y_test, 'prediccion' : y_pred})
   pickle.dump(reg, open("./model/modeldst.pkl","wb"))
   
   print( df_predicciones.head())
# def model_3(dataset):
#     print("say hello")
#     atributos = ['FECHADILI','PRECIOVTAX']
#     x = dataset[atributos]
#     x.index = x.FECHADILI
#     plt.rcParams['figure.figsize'] = (16, 9)
#     plt.style.use('fast')
     

#     x.set_index('FECHADILI', inplace=True)
#     x.index = pd.to_datetime(x.index)
#     x = x.sort_index()
    
#     print(x.index.min())
#     print(x.index.max())
#     print((x['2017']))
#     print((x['2018']))      
    
#     print(x.describe())
#     meses =x.resample('M').mean()
#     print(meses)

#     plt.plot(meses['2017'].values)
#     plt.plot(meses['2018'].values)

#     jusep2017 = x['2017-06-01':'2017-09-01']
#     plt.plot(jusep2017.values)
#     jusep2018 = x['2018-06-01':'2018-09-01']
#     plt.plot(jusep2018.values)
#     # load dataset
#     values = x.values
#     # ensure all data is float
#     values = values.astype('float32')
#     # normalize features
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
#     scaled = scaler.fit_transform(values)
#     reframed = series_to_supervised(scaled, PASOS, 1)
#     print(reframed.head())

# 	
#     # split into train and test sets
#     values = reframed.values
#     n_train_days = 315+289 - (30+PASOS)
#     print(n_train_days)
#     train = values[:n_train_days, :]
#     test = values[n_train_days:, :]
#     # split into input and outputs
#     x_train, y_train = train[:, :-1], train[:, -1]
#     x_val, y_val = test[:, :-1], test[:, -1]
#     # reshape input to be 3D [samples, timesteps, features]
#     x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
#     x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
#     print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
#         #series_to_supervised(x, n_in=1, n_out=1, dropnan=True)
# 	
#     EPOCHS=40
     
#     model = crear_modeloFF()
     
#     history=model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_val,y_val),batch_size=PASOS)
#     results=model.predict(x_val)
#     plt.scatter(range(len(y_val)),y_val,c='g')
#     plt.scatter(range(len(results)),results,c='r')
#     plt.title('validate')
#     plt.show()
     
# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = pd.DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
#     # put it all together
#     agg = pd.concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg
# 	
# def crear_modeloFF():
    
#     model = Sequential() 
#     model.add(Dense(PASOS, input_shape=(1,PASOS),activation='tanh'))
#     model.add(Flatten())
#     model.add(Dense(1, activation='tanh'))
#     model.compile(loss='mean_absolute_error',optimizer='Adam',metrics=["mse"])
#     model.summary()
#     return model  

def model4(dataset):
   
    #dataset = np.column_stack((dataset.data, dataset.target))
   # dataset = pd.DataFrame(dataset,columns = np.append(dataset.feature_names, "MEDV"))
    dataset.head(3)
    atributos = ['OB_FORMAL','AMPLIACION','ESTADO_ACT','ESTRATO','USO','FECHAINICIO','AREATOTZC','UNIDADESGA',
'PRECIOUNIG','TIPOVALOR','SIS_CONSTR','MANO_OBRAP','MANO_OBRAT','AREA_LOTE','AREAVENDIB','NUMUNIDEST','NUMUNIVEN',
'NUMUNIXVEN','NRO_EDIFIC','NRO_PISOS','TIPOVRDEST','AREATOTCO2','AREAVENUNI','TIPOVIVI','RANVIVI','Destino2']
    
    print("Randon Forest predict")
    
    x = dataset[atributos]
    y = dataset['PRECIOVTAX']
   
    X_train, X_test, y_train, y_test = train_test_split(
                                        x,
                                        y,
                                        random_state = 123
                                    )
        # Creación del modelo
    # ==============================================================================
    modelo = RandomForestRegressor(
                n_estimators =60,
                criterion    = 'mse',
                max_depth    = None,
                max_features = 'auto',
                oob_score    = False,
                n_jobs       = -1,
                random_state = 123
             )
    
    # Entrenamiento del modelo
    # ==============================================================================
    modelo.fit(X_train, y_train)
    
    

    # Error de test del modelo inicial
    # ==============================================================================
    predicciones = modelo.predict(X = X_test)

    df_predicciones = pd.DataFrame({'precio' : y_test, 'prediccion' : predicciones})
    print('Train: ', modelo.score(X_train,y_train))
    print('Test: ', modelo.score(X_test,y_test))
    print( df_predicciones.head())
    print(r2_score(y_test, predicciones))
    
  
    
  

    
  
    
  
    
    # Validación empleando el Out-of-Bag error
    # ==============================================================================
    train_scores = []
    oob_scores   = []
    
    # Valores evaluados
    estimator_range = range(1, 150, 5)
    
    # Bucle para entrenar un modelo con cada valor de n_estimators y extraer su error
    # de entrenamiento y de Out-of-Bag.
    for n_estimators in estimator_range:
        modelo = RandomForestRegressor(
                    n_estimators = n_estimators,
                    criterion    = 'mse',
                    max_depth    = None,
                    max_features = 'auto',
                    oob_score    = True,
                    n_jobs       = -1,
                    random_state = 123
                 )
        modelo.fit(X_train, y_train)
        train_scores.append(modelo.score(X_train, y_train))
        oob_scores.append(modelo.oob_score_)
        
    # Gráfico con la evolución de los errores
    fig, ax = plt.subplots(figsize=(6, 3.84))
    ax.plot(estimator_range, train_scores, label="train scores")
    ax.plot(estimator_range, oob_scores, label="out-of-bag scores")
    ax.plot(estimator_range[np.argmax(oob_scores)], max(oob_scores),
            marker='o', color = "red", label="max score")
    ax.set_ylabel("R^2")
    ax.set_xlabel("n_estimators")
    ax.set_title("Evolución del out-of-bag-error vs número árboles")
    plt.legend();
    print(f"Valor óptimo de n_estimators: {estimator_range[np.argmax(oob_scores)]}")

    # Validación empleando k-cross-validation y neg_root_mean_squared_error
    # ==============================================================================
    train_scores = []
    cv_scores    = []
    
    # Valores evaluados
    estimator_range = range(1, 150, 5)
    
    # Bucle para entrenar un modelo con cada valor de n_estimators y extraer su error
    # de entrenamiento y de k-cross-validation.
    for n_estimators in estimator_range:
        
        modelo = RandomForestRegressor(
                    n_estimators = n_estimators,
                    criterion    = 'mse',
                    max_depth    = None,
                    max_features = 'auto',
                    oob_score    = False,
                    n_jobs       = -1,
                    random_state = 123
                 )
        
        # Error de train
        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X = X_train)
        rmse = mean_squared_error(
                y_true  = y_train,
                y_pred  = predicciones,
                squared = False
               )
        train_scores.append(rmse)
        
        # Error de validación cruzada
        scores = cross_val_score(
                    estimator = modelo,
                    X         = X_train,
                    y         = y_train,
                    scoring   = 'neg_root_mean_squared_error',
                    cv        = 5
                 )
        # Se agregan los scores de cross_val_score() y se pasa a positivo
        cv_scores.append(-1*scores.mean())
        
    # Gráfico con la evolución de los errores
    fig, ax = plt.subplots(figsize=(6, 3.84))
    ax.plot(estimator_range, train_scores, label="train scores")
    ax.plot(estimator_range, cv_scores, label="cv scores")
    ax.plot(estimator_range[np.argmin(cv_scores)], min(cv_scores),
            marker='o', color = "red", label="min score")
    ax.set_ylabel("root_mean_squared_error")
    ax.set_xlabel("n_estimators")
    ax.set_title("Evolución del cv-error vs número árboles")
    plt.legend();
    print(f"Valor óptimo de n_estimators: {estimator_range[np.argmin(cv_scores)]}")

    # Validación empleando el Out-of-Bag error
    # ==============================================================================
    train_scores = []
    oob_scores   = []
    
    # Valores evaluados
    max_features_range = range(1, X_train.shape[1] + 1, 1)
    
    # Bucle para entrenar un modelo con cada valor de max_features y extraer su error
    # de entrenamiento y de Out-of-Bag.
    for max_features in max_features_range:
        modelo = RandomForestRegressor(
                    n_estimators = 60,
                    criterion    = 'mse',
                    max_depth    = None,
                    max_features = max_features,
                    oob_score    = True,
                    n_jobs       = -1,
                    random_state = 123
                 )
        modelo.fit(X_train, y_train)
        train_scores.append(modelo.score(X_train, y_train))
        oob_scores.append(modelo.oob_score_)
        
    # Gráfico con la evolución de los errores
    fig, ax = plt.subplots(figsize=(6, 3.84))
    ax.plot(max_features_range, train_scores, label="train scores")
    ax.plot(max_features_range, oob_scores, label="out-of-bag scores")
    ax.plot(max_features_range[np.argmax(oob_scores)], max(oob_scores),
            marker='o', color = "red")
    ax.set_ylabel("R^2")
    ax.set_xlabel("max_features")
    ax.set_title("Evolución del out-of-bag-error vs número de predictores")
    plt.legend();
    print(f"Valor óptimo de max_features: {max_features_range[np.argmax(oob_scores)]}")
    pickle.dump(modelo, open("./model/modelrandomf.pkl","wb"))

dataset=validate_data()
dataset=new_dataset(dataset)
Corre_data(dataset)
#model_1(dataset)
model_2(dataset)
##model=model_3(dataset)
model4(dataset)
    
#df_2 = dataset.apply(strip_obj, axis=0)