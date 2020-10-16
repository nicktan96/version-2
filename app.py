from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('heart.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("heart_prediction.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    data1 = request.form['a']
    if data1 == "":
        data1 = 5
    else:
        data1 = request.form['a']
    data2 = request.form['b']
    if data2 =="m" or data2 == "M":
        data2 = 0
    else:
        data2 = 1
        print(data2)

    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    arr = np.array([[data1, data2, data3,data4, data5, data6, data7, data8,data9]])
    prediction = model.predict(arr)

    output = prediction

    if output == 1:
        return render_template('heart_prediction.html', pred='High Risk of Heart Disease {}'.format(output))
    else:
        return render_template('heart_prediction.html', pred='Low Risk of Heart Disease {}'.format(output))



if __name__ == '__main__':
    app.run(debug=True)