
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

iris=datasets.load_iris()
print(iris.feature_names) # List of features (predictors)
print('X: ',iris.data[0:5]) # Firt 5 observations
print('y: ',np.unique(iris.target)) # List of unique values in the response variable

# Create the regressor matrix X
X=iris.data[:,0:2] # Selecting all rows for the first two columns/parameters
y=iris.target # 0: Setosa, 1: Versicolor, 2: Virginica

# Plot 
X_y0=X[y==0,:]
X_y1=X[y==1,:]
X_y2=X[y==2,:]
y0=y[y==0]
y1=y[y==1]
y2=y[y==2]

fig2D, ax2d = plt.subplots(figsize=(8,8))
ax2d.scatter(X_y0[:,0],X_y0[:,1], c='r', label='Setosa')
ax2d.scatter(X_y1[:,0],X_y1[:,1], c='b', label='Versicolor')
ax2d.scatter(X_y2[:,0],X_y2[:,1], c='k', label='Virginica')
ax2d.set_xlabel('Sepal length (cm)',fontsize = 'large', fontweight = 'bold')
ax2d.set_ylabel('Sepal width (cm)',fontsize = 'large', fontweight = 'bold')
ax2d.spines['right'].set_visible(False) 
ax2d.spines['top'].set_visible(False)
plt.legend()
plt.savefig('/Users/Documents/Iris project/Sepal_length_vs_width_2D.png')
plt.show()

fig3d = plt.figure(figsize=(8,8))
ax3d = plt.axes(projection = '3d')
plot1 = ax3d.scatter3D(X_y0[:,0], X_y0[:,1], y0, c='red', label = 'Setosa')
plot2 = ax3d.scatter3D(X_y1[:,0], X_y1[:,1], y1, c='blue' , label = 'Versicolor')
plot3 = ax3d.scatter3D(X_y2[:,0], X_y2[:,1], y2, c='black', label = 'Virginica')
ax3d.set_xlabel('Sepal length (cm)', fontsize ='large', fontweight = 'bold')
ax3d.set_ylabel('Sepal width (cm)',fontsize='large', fontweight='bold')
ax3d.set_zlabel('Iris types', fontsize='large', fontweight='bold')
ax3d.set_zlim(0,2)
major_zticks = np.arange(0, 3, 1)
ax3d.set_zticks(major_zticks)
plt.legend(loc = 2)
plt.savefig('/Users/Documents/Iris project/Sepal_length_vs_width_3D.png')
plt.show()
