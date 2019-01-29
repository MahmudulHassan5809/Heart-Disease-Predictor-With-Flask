from flask import Flask,render_template,url_for,request
from flask_material import Material

#EDA Pkg
import pandas as pd
import numpy as np

#ML Pkg
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pickle

app = Flask(__name__)
Material(app)


@app.route('/')
def index():
	return render_template("index.html")


@app.route('/preview')
def preview():
	df = pd.read_csv("data/heart.csv")
	return render_template("preview.html",df_view = df)

@app.route('/about')
def about():

	return render_template("about.html")

@app.route('/analyze',methods=['POST'])
def analyze():
	if request.method == 'POST':
		age = request.form['age']
		sex = request.form['sex']
		cp = request.form['cp']
		trestbps = request.form['trestbps']
		chol = request.form['chol']
		fbs = request.form['fbs']
		restecg = request.form['restecg']
		thalach = request.form['thalach']
		exang = request.form['exang']
		oldpeak = request.form['oldpeak']
		slope = request.form['slope']
		ca = request.form['ca']
		thal = request.form['thal']

		model_choice = request.form['model_choice']

		sample_data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

		# Clean the data by convert from unicode to float

		clean_data = [float(i) for i in sample_data]

		ex1 = [clean_data]

		df = pd.read_csv("data/heart.csv")
		feature_column_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
		predicted_class_name = ['target']

		# Getting feature variable values
		X = df[feature_column_names].values
		y = df[predicted_class_name].values

		# Saving 30% for testing
		split_test_size = 30

		# Splitting using scikit-learn train_test_split function
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_test_size, random_state = 42)

		#Impute with mean all 0 readings
		fill_0 = SimpleImputer(missing_values=0, strategy='mean')

		X_train = fill_0.fit_transform(X_train)
		X_test = fill_0.fit_transform(X_test)

		# Reloading the Model
		if model_choice == 'logitmodel':
		    logit_model = joblib.load('data/logit_model_heart2.pkl')
		    result_accuracy = logit_model.score(X_test, y_test)
		    result_prediction = logit_model.predict(ex1)

		elif model_choice == 'svmmodel':
			svm_model = joblib.load('data/svm_model_heart.pkl')
			result_accuracy = svm_model.score(X_test, y_test)
			result_prediction = svm_model.predict(ex1)
		elif model_choice == 'knnmodel':
			knn_model = joblib.load('data/knn_model_heart.pkl')
			result_accuracy = knn.score(X_test, y_test)
			result_prediction = knn_model.predict(ex1)
		elif model_choice == 'dtree':
			dtree_model = joblib.load('data/dtree_model_heart.pkl')
			result_accuracy = dtree_model.score(X_test, y_test)
			result_prediction = dtree_model.predict(ex1)
		elif model_choice == 'rrmodel':
			rr_model = joblib.load('data/rr_model_heart.pkl')
			result_accuracy = rr_model.score(X_test, y_test)
			result_prediction = rr_model.predict(ex1)
		elif model_choice == 'nbmodel':
			nb_model = joblib.load('data/nb_model_heart.pkl')
			result_accuracy = nb_model.score(X_test, y_test)
			result_prediction = nb_model.predict(ex1)

	return render_template("index.html",age = age ,
										sex = sex,
										cp = cp,
										trestbps = trestbps,
										chol = chol,
										fbs = fbs,
										restecg = restecg,
										thalach = thalach,
										exang =exang,
										oldpeak =oldpeak,
										slope =slope,
										ca =ca,
										thal =thal,
										result_prediction=result_prediction,
										model_selected=model_choice,
										result_accuracy = result_accuracy)


if __name__ == '__main__':
	app.run(debug=True)
