from trainers import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier

LR = SKLearnTrainer(LogisticRegression(n_jobs=-1))
LR.train()
LR.save()

SVM_LK = SKLearnTrainer(LinearSVC())
SVM_LK.train()
SVM_LK.save()

SVM_PK = SKLearnTrainer(SVC(kernel='poly'))
SVM_PK.train()
SVM_PK.save()

KNN = SKLearnTrainer(KNeighborsClassifier(n_jobs=-1))
KNN.train()
KNN.save()