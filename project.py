from flask import Flask, render_template, url_for, request
from sklearn.externals import joblib
import os, pickle


app = Flask(__name__, static_folder='static')

@app.route("/")
def index():
    return render_template('app.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
   
    sex = int(request.form['sex'])
    chol = float(request.form['chol'])
     age = int(request.form['age'])
    thalach = float(request.form['thalach'])
    exang = int(request.form['exang'])
    cp = int(request.form['cp'])
    x = np.array([sex, cp, chol, age, thalach, exang]).reshape(1, -1)

    path = os.path.join(os.path.dirname(__file__), 'models/scaler.pkl')
    scaler = None
    with open(path, 'rb') as f:
        scaler = pickle.load(f)

    x = scaler.transform(x)

    model_path = os.path.join(os.path.dirname(__file__), 'models/rfc.sav')
    clf = joblib.load(model_path)

    out = clf.predict(x)
    print(out)

    if out == 0:
        return render_template('not_having_disease.html')

    else:
        return render_template('heartdisease.htm', stage=int(out))


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
