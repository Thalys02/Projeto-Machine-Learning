import pandas as pd
data_frames = pd.read_csv('reclamacoes2012.csv')

X_df = data_frames[['Regiao','Tipo','SexoConsumidor']]

Y_df = data_frames[['Atendida']]

Xdummies_df = pd.get_dummies(X_df)

Ydummies_df = pd.get_dummies(Y_df)

X =  Xdummies_df.values         #Devolve os Dummies em Arrays

Y =  Y_df.values



#codigo para efetuar o treino e teste

tamanho_de_treino = int(0.9*len(Y))
tamanho_de_teste = len(Y)-tamanho_de_treino

treino_dados     = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados     =   X[-tamanho_de_teste:]
teste_marcacoes =   Y[-tamanho_de_teste:]



from sklearn.naive_bayes import  MultinomialNB

modelo = MultinomialNB()
modelo.fit(treino_dados,treino_marcacoes)

resultado = modelo.predict(teste_dados)
diferencia =  resultado - teste_marcacoes

acertos = [d for d in diferencia if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100.0 * total_de_acertos/total_de_elementos


print(taxa_de_acerto)
print(total_de_elementos)
