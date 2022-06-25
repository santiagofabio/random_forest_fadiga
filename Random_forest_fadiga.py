from copyreg import pickle
import pandas as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,  DecisionTreeRegressor
from sklearn.tree import _tree
from sklearn.metrics import accuracy_score,classification_report
import  matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from yellowbrick.classifier import ConfusionMatrix
from sklearn.ensemble import RandomForestClassifier
import pickle

#Random Forest
base_fadiga = np.read_excel('base_de_dados.xlsx')
print(f'{base_fadiga.head(3)}')
print(base_fadiga.describe())
print(f'{base_fadiga.columns }')

# Importando os atributos previsoes e classe ja tratados
with open('base_fadiga.pkl', 'rb') as f:
       x_fadiga_treinamento,x_fadiga_teste, y_fadiga_treinamento,y_fadiga_teste  = pickle.load(f)


print(f'Dimensões x_credito_treinamento:{x_fadiga_treinamento.shape}')
print(f'Dimensões y_credito_treinamento: {y_fadiga_treinamento.shape}')



print(f'Dimensões x_credito_teste:{x_fadiga_teste.shape}')
print(f'Dimensões y_credito_teste:{y_fadiga_teste.shape}')


random_forest_credito = RandomForestClassifier(n_estimators=20,criterion='entropy', random_state=0)
random_forest_credito.fit(x_fadiga_treinamento,y_fadiga_treinamento)
#n_estimators=10 -> Numéro de arvores a serem criadas(10) por padrão é 100
#criterion="gini", por default
#random_state=0, para dermos sempre os mesmo resultados.
# x_credito_treinamento -> parametros previsores 
# y_credito_treinamento -> Repostas 

previsoes = random_forest_credito.predict(x_fadiga_teste)
accuracy  = accuracy_score(y_fadiga_teste, previsoes)
print(f' Precisão {accuracy}')

cm = ConfusionMatrix(random_forest_credito)
cm.fit( x_fadiga_treinamento, y_fadiga_treinamento) 
score_cm = cm.score(x_fadiga_teste,y_fadiga_teste)
print(f'Score Confusion Matriz {score_cm}')
plt.savefig("matriz_de_confusao.png", dpi =300, format='png')
cm.show()




 

