import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.simplefilter("ignore")

def StringToInt(string):
	integer = 0
	try:
		integer = int(string)
	except:
		string = string.lower()
		for i in string:
			integer += ord(i)
	return integer


def prediction(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,
	Loan_Amount_Term,Credit_History,Property_Area):

	df = pd.read_csv(r"D:\django\Datasets\clustering.csv")
	df.head(10)

	df.info()
	df.describe()
	df.isnull().sum()

	df = df.drop("Loan_ID",axis =1)

	df.loc[df.Gender=="Male",'Gender'].shape
	df.Gender.mode()
	df.Gender= df.Gender.fillna("Male")

	df.Dependents.unique()
	df.Dependents.mode()
	df.Dependents = df.Dependents.fillna("0")

	df.Self_Employed.unique()
	df.Self_Employed.mode()
	df.Self_Employed = df.Self_Employed.fillna(method="ffill")

	df.Loan_Amount_Term.mode()
	df.Loan_Amount_Term.median()
	df.Loan_Amount_Term.mean()
	df.Loan_Amount_Term = df.Loan_Amount_Term.fillna(method="ffill")

	df.isnull().sum()

	df.Credit_History.unique()
	df.Credit_History = df.Credit_History.fillna(method="ffill")


	label = LabelEncoder()

	df.Gender = label.fit_transform(df.Gender)

	df.Married = label.fit_transform(df.Married)

	df.Dependents = label.fit_transform(df.Dependents)

	df.Education = label.fit_transform(df.Education)

	df.Self_Employed = label.fit_transform(df.Self_Employed)

	df.Property_Area = label.fit_transform(df.Property_Area)

	df.head()

	x = df.drop("Loan_Status",axis=1)
	y = df.Loan_Status

	scaler = StandardScaler()
	scaled = scaler.fit_transform(x)
	scaled

	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

	x_train.shape
	x_test.shape

	#Elbow Method


	Accuracy=[]

	for i in range(1,20):
	    knn = KNeighborsClassifier(n_neighbors=i)
	    
	    knn.fit(x_train,y_train)
	    yp = knn.predict(x_test)
	    Accuracy.append(accuracy_score(y_test,yp))

	Accuracy

	knn = KNeighborsClassifier(n_neighbors=9)
	knn.fit(x_train,y_train)

	yp = knn.predict(x_test)
	yp

	cm = confusion_matrix(y_test,yp)
	cm

	print(accuracy_score(y_test,yp))

	print(classification_report(y_test,yp))

	Gender = Gender.lower()
	Gender = StringToInt(Gender)

	Married = Married.lower()
	Married = StringToInt(Married)

	Dependents  =Dependents.lower()
	Dependents = StringToInt(Dependents)

	Education = Education.lower()
	Education = StringToInt(Education)

	Self_Employed = Self_Employed.lower()
	Self_Employed = StringToInt(Self_Employed)

	Property_Area = Property_Area.lower()
	Property_Area = StringToInt(Property_Area)

	ApplicantIncome = int(ApplicantIncome)
	CoapplicantIncome = int(CoapplicantIncome)
	Loan_Amount_Term = int(Loan_Amount_Term)
	LoanAmount = int(LoanAmount)
	Credit_History = int(Credit_History)


	return knn.predict([[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,
		LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]])