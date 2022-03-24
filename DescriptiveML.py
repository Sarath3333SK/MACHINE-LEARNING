import pandas as pd
class DescriptiveML():
    def init(self):
        pass
    def segreQuanQual(self,dataset):
        quantative=[]
        qualtative=[]

        for i in dataset.columns:
  
            if(dataset[i].dtypes =='object'):
                qualtative.append(i)
            else:
                quantative.append(i)
        print("The Quantitative Data:",quantative)
        print("The Qualtitative Data",qualtative)
        return quantative,qualtative
    def descriptive_Analysis(self,dataset,quantative):
        import pandas as pd
        des_data=pd.DataFrame(index=["Count","Mean","Median","Mode","Std","Min","Q1:25%","Q2:50%","Q3:75%","Q4:100%","IQR","1.5Rule",
                            "Lesser","Greater"],columns=quantative)

        for i in quantative:
            des_data[i]["Count"]=len(dataset[i])
            des_data[i]["Mean"]=dataset[i].mean()
            des_data[i]["Median"]=dataset[i].median()
            des_data[i]["Mode"]=dataset[i].mode()[0]
            des_data[i]["Std"]=dataset[i].describe()["std"]
            des_data[i]["Min"]=dataset[i].describe()["min"]
            des_data[i]["Q1:25%"]=dataset[i].describe()["25%"]
            des_data[i]["Q2:50%"]=dataset[i].describe()["50%"]
            des_data[i]["Q3:75%"]=dataset[i].describe()["75%"]
            des_data[i]["Q4:100%"]=dataset[i].describe()["max"]
            des_data[i]["IQR"]=des_data[i]["Q3:75%"]-des_data[i]["Q1:25%"]
            des_data[i]["1.5Rule"]=1.5* des_data[i]["IQR"]
            des_data[i]["Lesser"]= des_data[i]["Q1:25%"]-des_data[i]["1.5Rule"]
            des_data[i]["Greater"]= des_data[i]["Q3:75%"]+des_data[i]["1.5Rule"]

        return des_data
    def outliercolumn(self,quantative,des_data):
        lesser=[]
        greater=[]

        for i in quantative:
            if(des_data[i]["Lesser"]>des_data[i]['Min']):
                lesser.append(i)
            if(des_data[i]['Greater']<des_data[i]['Q4:100%']):
                greater.append(i)

        print("Lesser Range",lesser)
        print("Greater Range",greater)
        return lesser,greater

    def changeoutlier(self,dataset,des_Data,lesser,greater):
        for i in lesser:
            dataset[i][dataset[i]<des_Data[i]['Lesser']]=des_Data[i]['Lesser']
        #print(dataset[i])
        for j in greater:
            dataset[j][dataset[j]>des_Data[j]['Greater']]=des_Data[j]['Greater']
        return des_Data
    def random_forest(self,X_train, X_test, y_train, y_test):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        True_postive=cm[0][0]
        False_postive=cm[0][1]
        False_negative=cm[1][0]
        True_Negative=cm[1][1]
        Accuracy=(True_postive+True_Negative)/(True_postive+False_postive+False_negative+True_Negative)
        print("The Accuracy is:",Accuracy)
        return classifier,cm,Accuracy
    def Decision_Tree(self,X_train, X_test, y_train, y_test):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        True_postive=cm[0][0]
        False_postive=cm[0][1]
        False_negative=cm[1][0]
        True_Negative=cm[1][1]
        Accuracy=(True_postive+True_Negative)/(True_postive+False_postive+False_negative+True_Negative)
        print("The Accuracy is:",Accuracy)
        return classifier,cm,Accuracy
    def SVM_Linear(self,X_train, X_test, y_train, y_test):
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        True_postive=cm[0][0]
        False_postive=cm[0][1]
        False_negative=cm[1][0]
        True_Negative=cm[1][1]
        Accuracy=(True_postive+True_Negative)/(True_postive+False_postive+False_negative+True_Negative)
        print("The Accuracy is:",Accuracy)
        return classifier,cm,Accuracy
    def SVM_nonlinear(self,X_train, X_test, y_train, y_test):
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        True_postive=cm[0][0]
        False_postive=cm[0][1]
        False_negative=cm[1][0]
        True_Negative=cm[1][1]
        Accuracy=(True_postive+True_Negative)/(True_postive+False_postive+False_negative+True_Negative)
        print("The Accuracy is:",Accuracy)
        return classifier,cm,Accuracy
    def Knn(self,X_train, X_test, y_train, y_test):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        True_postive=cm[0][0]
        False_postive=cm[0][1]
        False_negative=cm[1][0]
        True_Negative=cm[1][1]
        Accuracy=(True_postive+True_Negative)/(True_postive+False_postive+False_negative+True_Negative)
        print("The Accuracy is:",Accuracy)
        return classifier,cm,Accuracy
    def NaiveBayes(self,X_train, X_test, y_train, y_test):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        True_postive=cm[0][0]
        False_postive=cm[0][1]
        False_negative=cm[1][0]
        True_Negative=cm[1][1]
        Accuracy=(True_postive+True_Negative)/(True_postive+False_postive+False_negative+True_Negative)
        print("The Accuracy is:",Accuracy)
        return classifier,cm,Accuracy