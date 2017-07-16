import pylab as p
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()



df = pd.read_csv('cars.csv')
df.Origin.value_counts()

CountStatus = df.Origin.value_counts()

CountStatus.plot.barh()


#df.plot(kind='bar', x= "Horsepower"), y = "Minutes Delayed.Weather")
X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)

p.plot(C,S)
p.show()
