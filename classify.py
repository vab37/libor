import pandas as pd
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import re
import sys

#print ('Number of arguments:', len(sys.argv), 'arguments.')
#print ('Argument List:', str(sys.argv))

filename = sys.argv[1]
#filename = './test/irswap_DEUTSCHE BANK_LIBOR_TEXT_edit.pdf'
modelloc = "./model/"

def get_input(filename):
    df = pd.DataFrame()
    f = open(filename, "r", encoding="utf8")
    d = {'file':str(filename),'txt':f.read() }
    df = df.append(d,ignore_index=True)
    return df

def doc_predict(df):
    loaded_model = pickle.load(open(modelloc+'model.pkl', 'rb'))
    loaded_transformer = pickle.load(open(modelloc+'transformer.pkl', 'rb'))
    X_test = df['txt']
    X_test_cv = loaded_transformer.transform(X_test)
    predictions = loaded_model.predict(X_test_cv)
    finaloutput = df.copy()#df.loc[:,['file']].copy()
    finaloutput['predictions']=predictions
    #finaloutput['class'] = finaloutput['predictions'].map({0:'mortgage', 1:'amendment',2:'normal',3:'term',4:'syndicate'})
    #finaloutput.loc[:, finaloutput.columns != 'txt'].to_csv(outputloc+'outputpredict.csv',index=False)
    return finaloutput




if __name__ == "__main__":
    df = get_input(filename)
    df.head()
    df = doc_predict(df)
    df.head()
    print(df.loc[0,'predictions'])
    pass


