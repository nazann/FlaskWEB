
from flask import Flask, render_template ,request,redirect, url_for, flash
import pickle
import numpy as np

app = Flask(__name__,template_folder='template')
wsgi_app = app.wsgi_app

# Make the WSGI interface available at the top level so wfastcgi can get it.


model=pickle.load(open('iri.pkl','rb'))
@app.route('/')
def man():
    return render_template('WebPage1.html')

@app.route('/predict',methods=['POST'])
def home():
    data1=request.form['a']
    data2=request.form['b']
    data3=request.form['c']
    data4=request.form['d']
    arr=np.array([[data1,data2,data3,data4]])
    pred=model.predict(arr)
    return render_template('afterpage.html',data=pred)



    
if __name__=="__main__":
    #app.run(debug=True)

    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)