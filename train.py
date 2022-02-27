import os
os.sys.path.append('/home/prathyusha/Projects/medical_categorization/')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
import pre_process
import pprint
from sklearn.model_selection import train_test_split


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import *
from sklearn.model_selection import *



random.seed(seed_value)
np.random.seed(seed_value)

data_dir = '/perfios/DATA/prathyusha/InsureTech/BERT/'
data = pd.read_csv(data_dir+'masterlist_prudential.csv')

def preprocess_(data):
    data = data.applymap(lambda x: ' '.join(str(x).split()))
    data['item_description_preprocessed'] = data['Item Description'].apply(lambda x: pre_process.text_preprocessing(str(x).upper()))
    
    return data


data = preprocess_(data)
dummy =  data[(data['Master Receipt Group']=='Doctor Charges')& (data['Master Receipt Item'] == 'Radiology')]
data = data.drop(dummy.index)


def l1_cat_cv(classifier, classifier_name):
    
    
    
    l1_clf = Pipeline(
    [('vect',
      TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=2500, token_pattern=r'[a-zA-Z]+')),
     ('clf', classifier)])
    
    whole_X = data['item_description_preprocessed']
    whole_Y = data['Master Receipt Group']
    x_train, x_test, y_train, y_test = train_test_split(whole_X, whole_Y, test_size=0.33, random_state=seed_value)
   


    l1_clf.fit(x_train,y_train)
    print(l1_clf.classes_)
    # predict class and prob values
    probabilities = l1_clf.predict_proba(x_test)

    order = np.argsort(probabilities, axis=1)
    classification1 = l1_clf.classes_[order[:, -1:]]
    classification1 = [item for sublist in classification1 for item in sublist]
    prob1 = probabilities[np.repeat(np.arange(order.shape[0]), 1), order[:, -1].flatten()].reshape(order.shape[0], 1)

    print('Accuracy: {0}'.format(accuracy_score(classification1, y_test)))
    print('Micro F1 score: {0}'.format(f1_score(classification1, y_test, average='micro')))
    print('Macro F1 score: {0}'.format(f1_score(classification1, y_test, average='macro')))
    print('Weighted F1 score: {0}'.format(f1_score(classification1, y_test, average='weighted')))
    # print('ROC_AUC score: {0}'.format(roc_auc_score(classification1, y_test)))

    df_confusion = pd.DataFrame(confusion_matrix(y_test, classification1, labels = l1_clf.classes_))

    plt.figure(figsize=(10,10))
    heatmap = sns.heatmap(df_confusion, annot=True, fmt="d", robust= True)
    heatmap.yaxis.set_ticklabels(l1_clf.classes_, rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(l1_clf.classes_, rotation=90, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    plt.savefig('CM.png')
    df_confusion.to_csv('CM_df.csv')

    clsf_report = pd.DataFrame(classification_report(y_true = y_test, y_pred = classification1, output_dict=True)).transpose()
    clsf_report.to_csv('CR.csv', index= True)
#     print(clsf_report.to_markdown())
    
  
randomForest = RandomForestClassifier(random_state=seed_value)
l1_cat_cv(randomForest,'RF')


