import matplotlib.pyplot as plt
import seaborn as sns

'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
'''
m = np.linspace(3,30,1000)
x = np.linspace(0.05,0.3,1000)

Freq = []
v = 200

for p in m:
    for r in x:
        f = 50*((1-r)/((1+p)**(1-r)-1))
        Freq.append(f)

df = pd.DataFrame(Freq)

import os

path = 'D:\data_HVSR'

folder = os.fsencode(path)

filenames = []
dataframe = []

Freq = pd.DataFrame(Freq)


for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith( ('.csv') ): # whatever file types you're using...
        filenames.append(filename)


for wcsv in filenames: 
    dataframe.append(pd.read_csv(wcsv,sep = '\t',names = ['Frequency', 'Amplitude'],index_col = False))


ampl = []
 
for t in range(0,len(m)):
    c = 0
    for d in dataframe:
        ad = d.iloc[t,1] + c
        c = ad 
    ampl.append(c/20)
    
ampl = pd.DataFrame(ampl)

   
#df = pd.read_excel('HVSR cal.xlsx',index_col = False)
#df.sort_values(["Freq"], axis=0, ascending=True, inplace=True)

dataset = pd.concat([dataframe[1],ampl],axis = 1)
dataset.drop(['Amplitude'],axis = 1, inplace = True)
#dataset1 = pd.read_excel('AverageHVSR.xlsx',index_col = False)

'''dataset_r = pd.DataFrame(dataset.round(3).values.tolist())
df_r = pd.DataFrame(df.round(3).values.tolist())

amp = []
index = []

for i in range(0,1000):
    for j in range(0,1000):
        if df_r.iloc[j,0] == dataset_r.iloc[i,0]:
            amp.append(dataset_r.iloc[i,1])
            
            index.append(j)
        else:
            continue
        
m = np.linspace(3,30,1000)
x = np.linspace(0.05,0.3,1000)

m_c = []
x_c = []

for q in index:
    m_c.append(m[q])
    x_c.append(x[q])'''
    
X = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1].values


'''from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)    
       
lin_reg_2.predict(poly_reg.fit_transform(np.array([df.iloc[1,0]]).reshape(1, 1)))

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')


plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')

Amplitude = []

for i in range(0,1000):
    Amplitude.append(lin_reg_2.predict(poly_reg.fit_transform(np.array([df.iloc[i,0]]).reshape(1, 1))))'''
 

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, y)     


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')   
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

y_pred = regressor.predict(np.array([df.iloc[1,0]]).reshape(1, 1))

Amplitude = []

for i in range(0,len(Freq)):
    Amplitude.append(regressor.predict(np.array([Freq.iloc[i,0]]).reshape(1, 1)))



X,Y = np.meshgrid(m,x)
z = np.array(Amplitude)
Z = z.reshape(len(m),len(m))

dfz = pd.DataFrame(Z)

dfz.to_excel_writer = "D:\data_HVSR/Book1.xlsx"
'''Amplitude = np.array(Amplitude)

m_d = pd.DataFrame(m)
x_d = pd.DataFrame(x)
Amplitude_d = pd.DataFrame(Amplitude)

df3=pd.DataFrame({ 'x':m, 'y': x, 'z':[Amplitude]*1000})

df1 = pd.concat([m_d,x_d,Amplitude_d], axis = 1,ignore_index=True)

z = Amplitude_d.values'''



%matplotlib inline
'''import cufflinks as cf
from plotly import __version__
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected = True)
cf.go_offline()'''
'''fig = df3.iplot(kind = 'surface')'''


'''fig = go.Figure(data = [go.surface(z = z,x = x, y = y)])
fig.update_layout(title = 'Mt Bruno Elevation', autosize = False, width = 500, height = 500, margin = dict(l=65,r = 50,b=65,t=90))

fig.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(m, x, Amplitude, c='y', marker='o')
ax.show()'''



#Z = np.array(df3['z'].values.tolist())
'''
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from plotly.graph_objs import *
init_notebook_mode()


trace0= Surface(m,x, Z)
data=[trace0]
fig=dict(data=data)
plot(fig)
'''
'''from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

from sys import argv

#x,y,z = np.loadtxt('your_file', unpack=True)

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(X, Y, Z)
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.savefig('teste.pdf')
plt.show()

from mpl_toolkits.mplot3d import Axes3D
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
ha.plot_surface(X, Y, Z)
plt.savefig('Surface.pdf')

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

fig = plt.figure(figsize = (10,10))
ax = fig.gca(projection = '3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X,Y,Z, rstride = 8, cstride = 8, alpha = 0.3)
cset = ax.contour(X, Y, Z, zdir = 'z', offset = -100,cmap = cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir = 'x', offset = -40,cmap = cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir = 'y', offset = -40,cmqp = cm.coolwarm)

plt.show()
plt.savefig('3D Contour.pdf')
'''
from mpl_toolkits.mplot3d import Axes3D
fig,ax = plt.subplots()
ax.contourf(X,Y,Z)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.contourf(X,Y,Z)

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.contour(X,Y,Z)


plt.show()

