from time import time
import pandas as pd
data_frames = pd.read_csv('reclamacoes2012.csv')


X_df = data_frames[['Regiao','Tipo','SexoConsumidor']]
Y_df = data_frames.Atendida

Y = list()

for i, value in enumerate(Y_df):
    if ( Y_df[i] == 'S'):
        Y.append(1)
    else:
        Y.append(0)

#TODO - TIRAR O Xdummies_df e fazer a mesma coisa q fiz

Xdummies_df = pd.get_dummies(X_df)

#Ydummies_df = pd.get_dummies(Y_df)

X =  Xdummies_df.values         #Devolve os Dummies em Arrays



#codigo para efetuar o treino e teste
from sklearn.model_selection import train_test_split
treino_dados, teste_dados, treino_marcacoes, teste_marcacoes = train_test_split(X, Y, test_size=0.1)

#print( treino_dados )
#print( treino_marcacoes )



from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import  MultinomialNB

clf_dict = {'log reg': LogisticRegression(),
            'naive bayes': GaussianNB(),
            'random forest': RandomForestClassifier(n_estimators=100),
            'knn': KNeighborsClassifier(),
            'linear svc': LinearSVC(),
            'ada boost': AdaBoostClassifier(n_estimators=100),
            'gradient boosting': GradientBoostingClassifier(n_estimators=100),
            'CART': DecisionTreeClassifier(),
            'Multinomial Naive Bayes': MultinomialNB()}

for name, clf in clf_dict.items():
    t0 = time()
    model = clf.fit(treino_dados, treino_marcacoes)
    pred = model.predict(teste_dados)
    model.score(teste_dados, teste_marcacoes)
    print('Precisão {}:'.format(name), model.score(teste_dados, teste_marcacoes))
    print("Tempo gasto:", round(time() - t0, 3), "s")

print("Fim da execução!")