
import numpy as np 
import pandas as pd 
import os
import pandas as pd
import itertools
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix # Loading required libraries

data = pd.read_csv(r"C:\Users\mahu123a\Documents\Data\RandomDataGeneration_Gendreau\RandomData_5_40_40/RandomData_5_40_40.csv")
data.dropna(inplace = True, axis = 0)

drop_cols = ["filename", "Route"]
labelcol = "CP Status"

y = data[labelcol]

y_true = y.sum()
y_false = len(y) - y_true

X= data.drop(columns = drop_cols + [labelcol])


X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3, 
                                                    random_state=42, stratify = y)

model =  XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    max_depth=10,
    n_estimators=300,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42,
    scale_pos_weight=y_true / y_false,
    n_jobs=1,            # keep model single-threaded
    verbosity=1,
    tree_method="hist"
)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(metrics.auc(y_test, y_pred))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#y_pred_classes = np.argmax(y_pred,axis = 1) 
confusion_mtx = confusion_matrix(y_test, y_pred) 
plot_confusion_matrix(confusion_mtx, classes = range(2)) 
# Again Plotting Confusion Matrix