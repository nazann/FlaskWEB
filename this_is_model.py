

from sklearn.svm import SVC
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


df= pd.read_csv('Iris - Copy.data')


X = np.array(df.iloc[:,0:4])
y = np.array(df.iloc[:,4])
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


sv=SVC(kernel='linear').fit(X_train,y_train)

#pred = svc_model.predict(X_test)
pickle.dump(sv,open('iri.pkl','wb'))