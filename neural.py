from sklearn.datasets import load_digits
mnist=load_digits()
X=mnist.data
y=mnist.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)



from sklearn.neural_network import MLPClassifier 
mlp=MLPClassifier(hidden_layer_sizes=(100,100,),activation="relu")
mlp.fit(X_train,y_train)

y_pred=mlp.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))