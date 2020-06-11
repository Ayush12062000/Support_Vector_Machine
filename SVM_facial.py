#%%
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    %matplotlib inline
    import time ,sys,os,warnings
    #for advance plot styling
    import seaborn as sns; sns.set()
    print("Libraries Imported Sucessfully")
except:
    print("Libraries Not Imported")

# %%
#generated Synthetic Dataset
from sklearn.datasets.samples_generator import make_blobs

X, y=make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)

plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='summer')

# %%
xfit = np.linspace(-1,3.5)
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap= 'summer')
plt.plot([0.6], [2.1], 'x', color='green', markeredgewidth=2, markersize=10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m*xfit+b, '-k')

plt.xlim(-1, 3.5)

# %%
#plotting the margins
xfit = np.linspace(-1,3.5)
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap= 'summer')

for m, b, d in [(1, 0.65,0.33), (0.5, 1.6, 0.55), (-0.2, 2.9,0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit-d, yfit+d, edgecolor='none', color='#AAAAAA', alpha=0.4)

plt.xlim(-1,3.5)

# %%
from sklearn.svm import SVC #Support vector classifier
model= SVC(kernel='linear', C=1E10)
model.fit(X, y)

#%%
def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P, colors='k', levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])

    # pLot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],model.support_vectors_[:, 1],s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim) 

# %%
plt.scatter(X[:,0],X[:,1],c=y, s=50, cmap='summer')
plot_svc_decision_function(model);

# %%
model.support_vectors_

# %%
#FACE RECOGNITION
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

# %%
fig , ax =plt.subplots(3, 4)
for i , axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap= 'bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])


# %%
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc =  SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca,svc)

# %%
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)
print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {'svc__C' : [1,5,10,50],'svc__gamma' : [0.0001, 0.0005,0.001,0.005]}

grid= GridSearchCV(model , param_grid)
%time grid.fit(Xtrain, ytrain)
print(grid.best_params_)

# %%
model  = grid.best_estimator_
yfit = model.predict(Xtest)

# %%
fig , ax = plt.subplots(4,6)
for i ,axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62,47),cmap='bone')
    axi.set(xticks=[],yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],color='black' if yfit[i]==ytest[i] else 'red')

fig.suptitle('Predicted Names; Incorrect Labels in Red',size=15)

# %%
from sklearn.metrics import classification_report , confusion_matrix

print(classification_report(ytest , yfit , target_names= faces.target_names))

#%%
mat = confusion_matrix(ytest , yfit)

sns.heatmap(mat.T , square=True, annot= True , fmt='d', cbar= True,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('True label');
plt.ylabel('Predicted label');

# %%
print("SCORE:{:0.2f}".format(model.score(Xtest,ytest)))


# %%
