#Clase de modelos predictivos para series de tiempo

import warnings
import statistics
import pandas as pd
from abc import  ABCMeta, abstractmethod
from statsmodels.tsa.api import ExponentialSmoothing
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
#import PCA
#from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import AdaBoostRegressor


from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge


class BasePrediccion(metaclass = ABCMeta):    
  @abstractmethod
  def forecast(self):
    pass

class Prediccion(BasePrediccion):
  def __init__(self, modelo):
    self.__modelo = modelo
  
  @property
  def modelo(self):
    return self.__modelo  
  
  @modelo.setter
  def modelo(self, modelo):
    if(isinstance(modelo, Modelo)):
      self.__modelo = modelo
    else:
      warnings.warn('El objeto debe ser una instancia de Modelo.')
  
class meanfPrediccion(Prediccion):
  def __init__(self, modelo):
    super().__init__(modelo)
  
  def forecast(self, steps = 1):
    res = []
    for i in range(steps):
      res.append(self.modelo.coef)
    
    start = self.modelo.ts.index[-1]
    freq  = self.modelo.ts.index.freqstr
    fechas = pd.date_range(start = start, periods = steps+1, freq = freq)
    fechas = fechas.delete(0)
    res = pd.Series(res, index = fechas)
    return(res)

class naivePrediccion(Prediccion):
  def __init__(self, modelo):
    super().__init__(modelo)
  
  def forecast(self, steps = 1):
    res = []
    for i in range(steps):
      res.append(self.modelo.coef)
    
    start = self.modelo.ts.index[-1]
    freq  = self.modelo.ts.index.freqstr
    fechas = pd.date_range(start = start, periods = steps+1, freq = freq)
    fechas = fechas.delete(0)
    res = pd.Series(res, index = fechas)
    return(res)

class snaivePrediccion(Prediccion):
  def __init__(self, modelo):
    super().__init__(modelo)
  
  def forecast(self, steps = 1):
    res = []
    pos = 0
    for i in range(steps):
      if pos >= len(self.modelo.coef):
        pos = 0
      res.append(self.modelo.coef[pos])
      pos = pos + 1
    
    start = self.modelo.ts.index[-1]
    freq  = self.modelo.ts.index.freqstr
    fechas = pd.date_range(start = start, periods = steps+1, freq = freq)
    fechas = fechas.delete(0)
    res = pd.Series(res, index = fechas)
    return(res)

class driftPrediccion(Prediccion):
  def __init__(self, modelo):
    super().__init__(modelo)
  
  def forecast(self, steps = 1):
    res = []
    for i in range(steps):
      res.append(self.modelo.ts[-1] + self.modelo.coef * i)
    
    start = self.modelo.ts.index[-1]
    freq  = self.modelo.ts.index.freqstr
    fechas = pd.date_range(start = start, periods = steps+1, freq = freq)
    fechas = fechas.delete(0)
    res = pd.Series(res, index = fechas)
    return(res)

class BaseModelo(metaclass = ABCMeta):    
  @abstractmethod
  def fit(self):
    pass

class Modelo(BaseModelo):
  def __init__(self, ts):
    self.__ts = ts
    self._coef = None
  
  @property
  def ts(self):
    return self.__ts  
  
  @ts.setter
  def ts(self, ts):
    if(isinstance(ts, pd.core.series.Series)):
      if(ts.index.freqstr != None):
        self.__ts = ts
      else:
        warnings.warn('ERROR: No se indica la frecuencia de la serie de tiempo.')
    else:
      warnings.warn('ERROR: El parámetro ts no es una instancia de serie de tiempo.')
  
  @property
  def coef(self):
    return self._coef
  
class meanf(Modelo):
  def __init__(self, ts):
    super().__init__(ts)
  
  def fit(self):
    self._coef = statistics.mean(self.ts)
    res = meanfPrediccion(self)
    return(res)

class naive(Modelo):
  def __init__(self, ts):
    super().__init__(ts)
  
  def fit(self):
    self._coef = self.ts[-1]
    res = naivePrediccion(self)
    return(res)

class snaive(Modelo):
  def __init__(self, ts):
    super().__init__(ts)
  
  def fit(self, h = 1):
    self._coef = self.ts.values[-h:]
    res = snaivePrediccion(self)
    return(res)

class drift(Modelo):
  def __init__(self, ts):
    super().__init__(ts)
  
  def fit(self):
    self._coef = (self.ts[-1] - self.ts[0]) / len(self.ts)
    res = driftPrediccion(self)
    return(res)



# Holt-Winters calibrado

#Clase de calibración de modelo

class HW_Prediccion(Prediccion):
  def __init__(self, modelo, alpha, beta, gamma):
    super().__init__(modelo)
    self.__alpha = alpha
    self.__beta  = beta
    self.__gamma = gamma
  
  @property
  def alpha(self):
    return self.__alpha
  
  @property
  def beta(self):
    return self.__beta 
  
  @property
  def gamma(self):
    return self.__gamma
  
  def forecast(self, steps = 1):
    res = self.modelo.forecast(steps)
    return(res)
  
class HW_calibrado(Modelo):
  def __init__(self, ts, test, trend = 'add', seasonal = 'add'):
    super().__init__(ts)
    self.__test = test
    self.__modelo = ExponentialSmoothing(ts, trend = trend, seasonal = seasonal)
  
  @property
  def test(self):
    return self.__test  
  
  @test.setter
  def test(self, test):
    if(isinstance(test, pd.core.series.Series)):
      if(test.index.freqstr != None):
        self.__test = test
      else:
        warnings.warn('ERROR: No se indica la frecuencia de la serie de tiempo.')
    else:
      warnings.warn('ERROR: El parámetro ts no es una instancia de serie de tiempo.')
  
  def fit(self, paso = 0.1):
    error = float("inf")
    n = np.append(np.arange(0, 1, paso), 1)
    for alpha in n:
      for beta in n:
        for gamma in n:
          model_fit = self.__modelo.fit(smoothing_level = alpha, smoothing_trend = beta, smoothing_seasonal = gamma)
          pred      = model_fit.forecast(len(self.test))
          mse       = sum((pred - self.test)**2)
          if mse < error:
            res_alpha = alpha
            res_beta  = beta
            res_gamma = gamma
            error = mse
            res = model_fit
    return(HW_Prediccion(res, res_alpha, res_beta, res_gamma))


# Clase de errores

import math
from numpy import corrcoef

class ts_error:
  def __init__(self, preds, real, nombres = None):
    self.__preds = preds
    self.__real = real
    self.__nombres = nombres
  
  @property
  def preds(self):
    return self.__preds
  
  @preds.setter
  def preds(self, preds):
    if(isinstance(preds, pd.core.series.Series) or isinstance(preds, numpy.ndarray)):
      self.__preds = [preds]
    elif(isinstance(preds, list)):
      self.__preds = preds
    else:
      warnings.warn('ERROR: El parámetro preds debe ser una serie de tiempo o una lista de series de tiempo.')
  
  @property
  def real(self):
    return self.__real
  
  @real.setter
  def real(self, real):
    self.__real = real
  
  @property
  def nombres(self):
    return self.__nombres
  
  @nombres.setter
  def nombres(self, nombres):
    if(isinstance(nombres, str)):
      nombres = [nombres]
    if(len(nombres) == len(self.__preds)):
      self.__nombres = nombres
    else:
      warnings.warn('ERROR: Los nombres no calzan con la cantidad de métodos.')
  
  def RSS(self):
    res = []
    for pred in self.preds:
      res.append(sum((pred - self.real)**2))
    return(res)
  
  def MSE(self):
    return([pred / len(self.real) for pred in self.RSS()])
  
  def RMSE(self):
    return([math.sqrt(pred) for pred in self.MSE()])
  

  
  def RE(self):
    res = []
    for pred in self.preds:
      try:
          res.append(sum(abs(self.real - pred)) / sum(abs(self.real)))
      except:
          ("Valor 0")
    return(res)
  
  def CORR(self):
    res = []
    for pred in self.preds:
      corr = corrcoef(self.real, pred)[0, 1]
      res.append(0 if math.isnan(corr) else corr)
    return(res)
  
  def MAE(self):
    res = []
    for pred in self.preds:
        res.append(sum(abs(self.real - pred)) / len(self.real))
    return(res)

  
  def df_errores(self):
    res = pd.DataFrame({'MAE': self.MAE(),'MSE': self.MSE(), 'RMSE': self.RMSE(), 'RE': self.RE(), 'CORR': self.CORR()})
    if(self.nombres is not None):
      res.index = self.nombres
    return(res)
  
  def __escalar(self):
    res = self.df_errores()
    for nombre in res.columns.values:
      res[nombre] = res[nombre] - min(res[nombre])
      res[nombre] = res[nombre] / max(res[nombre]) * 100
    return(res)
  
  def plot_errores(self):
    plt.figure(figsize=(8, 8))
    df = self.__escalar()
    if(len(df) == 1):
      df.loc[0] = 100
    
    N = len(df.columns.values)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    
    ax = plt.subplot(111, polar = True)
    
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], df.columns.values)
    
    ax.set_rlabel_position(0)
    plt.yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], color = "grey", size = 10)
    plt.ylim(-10, 110)
    
    for i in df.index.values:
      p = df.loc[i].values.tolist()
      p = p + p[:1]
      ax.plot(angles, p, linewidth = 1, linestyle = 'solid', label = i)
      ax.fill(angles, p, alpha = 0.1)
    
    plt.legend(loc = 'best')
    plt.show()
    
  def plotly_errores(self):
    df = self.__escalar()
    etqs = df.columns.values.tolist()
    etqs = etqs + etqs[:1]
    if(len(df) == 1):
      df.loc[0] = 100
    
    fig = go.Figure()
    
    for i in df.index.values:
      p = df.loc[i].values.tolist()
      p = p + p[:1]
      fig.add_trace(go.Scatterpolar(
        r = p, theta = etqs, fill = 'toself', name = i
      ))
    
    fig.update_layout(
      polar = dict(
        radialaxis = dict(
          visible = True,
          range=[-10, 110]
      ))
    )
    
    return(fig)
  


#clase que encuentra la mejor correlación, desfase y periodo para 2 series de tiempo


class find_best_lag():
    def __init__(self, ts,  ts1):
      self.__ts = ts
      self.__ts1 = ts1
    
    @property
    def ts(self):
        return self.__ts
    
    @ts.setter
    def ts(self, ts):
        self.__ts = ts  
    
       
    @property
    def ts1(self):
        return self.__ts1
    
    @ts1.setter
    def ts1(self, ts1):
        self.__ts1 = ts1
        
    def best_lag(self):
        max_corr = 0
        best_lag = 0
        max_corr_ts = None
        mejor_periodo = None
        
        for periodo in range(25): #rango de desfase a probar 
            # Desplaza la serie ts2
            #super().__init__(self.ts, )
            #Acumulacion.periodo = periodo
            shifted_ts = self.__ts.rolling(periodo).sum().dropna()
            #recorte de las series para que tengan misma longitud
            ts1= self.__ts1.loc [self.__ts1.index <= shifted_ts.index.max()]
            ts1= self.__ts1.loc [self.__ts1.index >= shifted_ts.index.min()]
            shifted_ts = shifted_ts.loc[shifted_ts.index >= ts1.index.min()]
            shifted_ts = shifted_ts.loc[shifted_ts.index <= ts1.index.max()]
            #corr = ts1.corr(shifted_ts)
            
            for lag in range(5,20): #rango de desfases
              shifted_ts2 = shifted_ts.shift(lag).dropna()
              corr = ts1.corr(shifted_ts2)
              
              if abs(corr) > abs(max_corr):
                max_corr = corr
                best_lag = lag
                max_corr_ts = shifted_ts2
                mejor_periodo = periodo
                
        
        return best_lag, max_corr, max_corr_ts, ts1, mejor_periodo, self.__ts

### Clase que crea un df con los parameteros de mejor desfase, periodo y variable de clima

class creardf (find_best_lag ):
    def __init__(self, ts,  ts1, nueva_col):
        find_best_lag.__init__(self, ts, ts1)      
        self.__nueva_col = nueva_col
        

    
    @property    
    def nueva_col(self):
        return self.__nueva_col
    nueva_col.setter
    def nueva_col(self, nueva_col):
        self.__nueva_col = nueva_col
          
    def crear_df(self):
        best_lag, max_corr, max_corr_ts, ts1, mejor_periodo, ts = self.best_lag()
        print ("Mejor desfase: ", best_lag,
             "\nMejor periodo: ",mejor_periodo,)
        print ("Mejor correlación: ", max_corr)
        max_corr_ts = max_corr_ts.rename(self.__nueva_col)

        #Crear df con ts de max correlacion
        df = pd.DataFrame(max_corr_ts)

        #Se agrega al df la serie del fenomeno 
        df = df.merge(ts1, left_index=True, right_index=True)
        
        #llamando a la serie orginal con los datos de clima de la variable
        #serie_ori = find_best_lag.ts
        ts = ts.rename("variable_clima")
        df = df.merge(ts, left_index=True, right_index=True)
        
        
        fechas = pd.DatetimeIndex(df.index)
        df["fechas"] = fechas
        #Relleno el df con la fecha maxima que se puede según el desfase y ultima fecha de clima
        fecha_ul = df.index.max().date() + timedelta(weeks =best_lag)
        total_fechas  = pd.date_range(start = df.index.max().date(), end = fecha_ul, freq = 'W')
        faltan_fechas = [x for x in total_fechas if x not in df.fechas.to_list()]
        
        df = pd.concat([df, pd.DataFrame({'fechas': faltan_fechas })], ignore_index = True)
        df = df.sort_values(by = ['fechas'])
        df.index = df['fechas']
        
        #iteracion para rellenar las fechas 
        for i in faltan_fechas: 
            acumulacion = df.loc[(df.index <= (i -timedelta(weeks=best_lag)) ) & (df.index > (i-timedelta(weeks=best_lag+mejor_periodo)))]["variable_clima"].sum()
            df.loc[i, self.__nueva_col] = acumulacion

        

        return df
    






### Periodo critico 

class periodo_critico(find_best_lag):
    def __init__(self, ts, periodo, ts1, variable_clima, nueva_col):
        super().__init__(ts, periodo, ts1)
        self.__variable_clima = variable_clima
        self.__nueva_col = nueva_col

    
    @property
    def df(self):
        return self.__df
    @df.setter
    def df(self, df):
        self.__df = df

    @property
    def nueva_col(self):
        return self.__nueva_col
    @nueva_col.setter
    def nueva_col(self,nueva_col):
        self.__nueva_col = nueva_col
    
    
    @property
    def periodo(self,):
        return (self.__periodo)
    @periodo.setter
    def periodo(self, periodo):
        self.__periodo = periodo
    @property
    def desfase(self):
        return (self.__desfase)
    @desfase.setter
    def desfase(self, desfase):
        self.__desfase = desfase
    
    @property
    def variable_clima(self,):
        return (self.__variable_clima)
    @variable_clima.setter
    def variable_clima(self,variable_clima):
        self.__variable_clima = variable_clima
    
    def calcula_clima(self):
        
        # Convertimos a serie de tiempo.
        fechas = pd.DatetimeIndex(self.__df.index, dtype='datetime64[ns]', freq='W')
        ts_W= pd.Series(self.__df [self.__variable_clima].values, index = fechas)

        acumulacion = ts_W.rolling(self.__periodo).sum()
        #desfase = acumulacion.shift(self.__desfase)
        
        acumulacion = acumulacion.shift(periods = self.__desfase)
        
        
        # Calcular el valor en segundos
        #segundos_a_restar = self.__desfase * 604800
        
        #acumulacion.index = acumulacion.index - pd.Timedelta(seconds= segundos_a_restar)
        
        acumulacion =acumulacion.rename(self.__nueva_col)
        df = self.__df 
       
        df= df.merge(acumulacion, left_index=True, right_index=True)
        
        return (df)



#Clase de modelos de prediccion de ML
  
class Modelos:
    def __init__(self, X_train, Y_train,X_test, y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.y_test = y_test
        self.X_test = X_test

    def eval_modelos(self) :
        df_ts = pd.DataFrame()
        models = [LinearRegression(), RandomForestRegressor(), SVR(), GradientBoostingRegressor(), BayesianRidge()]
        # MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
        #self.y_test = y_test
        results = {}
        df_results = pd.DataFrame()
        #fechas = pd.DatetimeIndex(df_test.index)

        for model in models:
            model.fit(self.X_train, self.Y_train)
            predictions = model.predict(self.X_test)
            predictions[predictions < 0] = 0
            mae = mean_absolute_error(self.y_test, predictions)
            mse = mean_squared_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)
            RMSE = metrics.mean_squared_error(self.y_test, predictions, squared=False)
            results[str(model)] = [mae, mse, RMSE,r2]
            df_ts[str(model)] = predictions.tolist()
            df_results = pd.DataFrame.from_dict(results, orient='index', columns=['MAE', 'MSE', "RMSE","r2"])


        return (df_results, df_ts)
      
      

#Clase para buscar fechas faltantes 

class fechas_faltantes():
  def __init__(self, df,campo_fecha, campo_para_ts, ): 
    self.__campo_fecha = campo_fecha
    self.__df = df
    self.__campo_para_ts= campo_para_ts
    
    
  @property
  def campo_para_ts(self):
    return self.__campo_para_ts
  @campo_para_ts.setter 
  def campo_para_ts(self,campo):
    self.__campo_para_ts = campo
  
    
    
  @property
  def df(self):
    return self.__df
  @df.setter
  def df(self, df):
    self.__df = df
    
  @property
  def campo_fecha(self):
    return self.__campo_fecha
  @campo_fecha.setter
  def campo_fecha(self, campo_fecha):
    self.__campo_fecha = campo_fecha
    
  def crear_ts(self):
    
    #Buscamos cuales son las fechas faltantes.

    fechas = pd.DatetimeIndex(self.__df[self.__campo_fecha])
    df = self.__df
    df["fechas"] = fechas
    fecha_inicio = df["fechas"].to_list()[0]
    fecha_final  = df.fechas.to_list()[len(df.fechas.to_list()) - 1]

    total_fechas = pd.date_range(start = fecha_inicio, end = fecha_final, freq = 'D').tolist()
    faltan_fechas = [x for x in total_fechas if x not in df.fechas.to_list()]
    print ("faltan: " , len(faltan_fechas), " fechas")
    
    #Unimos y ordenamos las fechas faltantes.
    
    df = pd.concat([df, pd.DataFrame({'fechas': faltan_fechas})], ignore_index = True)
    df = df.sort_values(by = ['fechas'])
    df.index = df['fechas']
    
    #remover duplicados 
    df= df[~df.index.duplicated(keep='first')]
    
    #crear ts 
    fechas = pd.DatetimeIndex(df.index, dtype='datetime64[ns]', freq='D')
    df_ts = pd.Series(df[self.__campo_para_ts].values, index = fechas)
    print ("Total de ts: ", len (df_ts))
    print ("Inicia: " , fecha_inicio.date() )
    print ( "Fin: ", fecha_final.date())
    df_ts = df_ts.rename(self.__campo_para_ts)
    
    #df_new = df_ts.to_frame(name=f"{str(self.__campo_para_ts)}AC")
    
    #df_output = df_new.merge ()
    
    return df_ts
  

