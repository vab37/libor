import slate3k as slate
import pandas as pd
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import re
import sys
import warnings

#print ('Number of arguments:', len(sys.argv), 'arguments.')
#print ('Argument List:', str(sys.argv))

filename = sys.argv[1]
#filename = './test/irswap_DEUTSCHE BANK_LIBOR_TEXT_edit.pdf'
modelloc = "./model/"

def get_input(filename):
    df = pd.DataFrame()
    file = filename
    if file.endswith(".pdf"):
        #pdfFileObj = open(fileloc+"/"+file, 'rb') 
        #pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
        #print(pdfReader.numPages) 
        with open(filename, 'rb') as f:
            pages = slate.PDF(f)
        page_content=""                # define variable for using in loop.
        search_word = "LIBOR"
        search_word2 = '(libor unavailability) | (unavailability of libor)'
        #search_word2 = '(deutsche bank) | (interest rate)'
        search_word_count = 0
        startpos = -1
        startpos2 = -1
        startpage = 0
        pagepos = -1
        pagepos2 = -1
        startpagepos = -1
        startpage2 = 0
        startpagepos2 = -1
        snippet = ""
        snippet2 = ""
        hasfallback = "N"
        snippetlen = 30
        for page_number in range(len(pages)):
            #page = pdfReader.getPage(page_number)
            text = pages[page_number]#page.extractText()#.encode('utf-8')
            pagepos = text.lower().find(search_word.lower())
            if re.search(search_word2,text.lower()) is not None:
                pagepos2 = re.search(search_word2,text.lower()).start()
            
            if pagepos!=-1:
                search_word_count += 1
                if(startpage==0):
                    startpage = page_number+1
                    startpagepos=pagepos
            if pagepos2!=-1:
                if(startpage==0):
                    startpage2 = page_number+1
                    startpagepos2=pagepos2

            page_content += text
        
        startpos = page_content.find(search_word)
        if re.search(search_word2,page_content.lower()) is not None:
            startpos2 = re.search(search_word2,page_content.lower()).start()

        if startpos>=0:
            if(startpos-snippetlen >=0):
                snippet = page_content[startpos-snippetlen:startpos+snippetlen]
            else:
                snippet = page_content[0:startpos+snippetlen]
        if startpos2>=0:
            hasfallback = "Y"
            if(startpos2-snippetlen >=0):
                snippet2 = page_content[startpos2-snippetlen:startpos2+snippetlen]
            else:
                snippet2 = page_content[0:startpos2+snippetlen]


        d = {'file':str(file),'txt':page_content,'LIBOR_startpage':startpage,'LIBOR_startpageposition':startpagepos,'LIBOR_docposition':startpos,'LIBOR_startsnippet':snippet,'fallbackPresent':hasfallback,'fallbackPosition':startpos2,'fallbackPage':startpage2,'fallbackText':snippet2,'fallbackTextComplexity':0 }
        
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


