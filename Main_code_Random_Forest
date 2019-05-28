import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

iris=datasets.load_iris()
print(iris.feature_names) # List of features (predictors)
print('X: ',iris.data[0:5]) # Firt 5 observations
print('y: ',np.unique(iris.target)) # List of unique values in the response variable

# Create the regressor matrix X
X=iris.data[:,0:5] # Selecting all rows for the first two columns/parameters
y=iris.target # 0: Setosa, 1: Versicolor, 2: Virginica

# Plottiing only wrt sepal length and width
X_y0 = X[y == 0,:]
X_y1 = X[y == 1,:]
X_y2 = X[y== 2,:]
y0 = y[y == 0]
y1 = y[y == 1]
y2 = y[y == 2]

fig2D, ax2d = plt.subplots(figsize=(8,8))
ax2d.scatter(X_y0[:,0],X_y0[:,1], c='r', label='Setosa')
ax2d.scatter(X_y1[:,0],X_y1[:,1], c='b', label='Versicolor')
ax2d.scatter(X_y2[:,0],X_y2[:,1], c='k', label='Virginica')
ax2d.set_xlabel('Sepal length (cm)',fontsize = 'large', fontweight = 'bold')
ax2d.set_ylabel('Sepal width (cm)',fontsize = 'large', fontweight = 'bold')
ax2d.spines['right'].set_visible(False) 
ax2d.spines['top'].set_visible(False)
plt.legend()
plt.savefig('/Users/Iris project/Sepal_length_vs_width_2D.png')
plt.show()

#fig3d = plt.figure(figsize=(8,8))
#ax3d = plt.axes(projection = '3d')
#plot1 = ax3d.scatter3D(X_y0[:,0], X_y0[:,1], y0, c='red', label = 'Setosa')
#plot2 = ax3d.scatter3D(X_y1[:,0], X_y1[:,1], y1, c='blue' , label = 'Versicolor')
#plot3 = ax3d.scatter3D(X_y2[:,0], X_y2[:,1], y2, c='black', label = 'Virginica')
#ax3d.set_xlabel('Sepal length (cm)', fontsize ='large', fontweight = 'bold')
#ax3d.set_ylabel('Sepal width (cm)',fontsize='large', fontweight='bold')
#ax3d.set_zlabel('Iris types', fontsize='large', fontweight='bold')
#ax3d.set_zlim(0,2)
#major_zticks = np.arange(0, 3, 1)
#ax3d.set_zticks(major_zticks)
#plt.legend(loc = 'upper left')
#plt.savefig('/Users/Iris project/Sepal_length_vs_width_3D.png')
#plt.show()


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
tree_size = list(np.arange(1,100,5))
cv_accuracy = []
for i in tree_size:
    model_rf = RandomForestClassifier(n_estimators = i)
    model_accuracy=cross_val_score(model_rf, X, y, cv = 5, scoring='accuracy')  
#    #https://scikit-learn.org/stable/modules/model_evaluation.html
    cv_accuracy.append(100 * model_accuracy.mean()) # mean of 5-folds, 100 %
#    
#    model_rf.fit(X_train,y_train)
#    y_pred=model_rf.predict(X_test)
#    cv_accuracy.append(100 * metrics.accuracy_score(y_test, y_pred)) # 100%
    
max_accuracy = max(cv_accuracy)
tree_size_optimal = tree_size[cv_accuracy.index(max_accuracy)]


fig_rf, ax_rf = plt.subplots(figsize=(6,6))
ax_rf.plot(tree_size, cv_accuracy, label = 'Model accuracy')
ax_rf.set_xlabel('Tree size', fontsize='large', fontweight='bold')
ax_rf.set_ylabel('Model accuracy (%)', fontsize='large', fontweight='bold')
ax_rf.set_title('Methods: Random forest and k-fold cross-validation', fontsize='large', fontweight='bold')
ax_rf.scatter(tree_size_optimal, max_accuracy, c = 'r', label = 'Highest accuracy')
ax_rf.legend(loc = 'center right')
ax_rf.spines['right'].set_visible(False) 
ax_rf.spines['top'].set_visible(False)
ax_rf.set_ylim(75,100)
plt.text(tree_size_optimal, max_accuracy,str(round(max_accuracy, 2)),fontsize=12,fontweight='bold',
                    ha='left',va='bottom',color='yellow',
                    bbox=dict(facecolor='b', alpha=0.2))
plt.savefig('/Users/RandomForest_and_kfoldCV.png')
plt.show()
