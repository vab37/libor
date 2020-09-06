
import PyPDF2 
import pandas as pd
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
trainloc = "./train/"
testloc = "./test/"
outputloc = "./output/"
modelloc = "./model/"


def get_input(fileloc,filetype):
    df = pd.DataFrame()
    for file in os.listdir(fileloc):
        #print(file)
        if file.endswith(".pdf"):
            #print(os.path.join(fileloc, file))
            #print(fileloc+"/"+file)
            #print(file.split('_')[0])
            pdfFileObj = open(fileloc+"/"+file, 'rb') 
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
            print(pdfReader.numPages) 
            page_content=""                # define variable for using in loop.
            search_word = "LIBOR"
            search_word_count = 0
            startpos = 0
            startpage = 0
            startpagepos = -1
            snippet = ""
            snippetlen = 30
            for page_number in range(pdfReader.numPages):
                page = pdfReader.getPage(page_number)
                text = page.extractText()#.encode('utf-8')
                pagepos = text.lower().find(search_word.lower())
                if pagepos!=-1:
                    search_word_count += 1
                    if(startpage==0):
                        startpage = page_number+1
                        startpagepos=pagepos

                page_content += text
            
            startpos = page_content.find(search_word)
            if startpos>=0:
                if(startpos-snippetlen >=0):
                    snippet = page_content[startpos-snippetlen:startpos+snippetlen]
                else:
                    snippet = page_content[0:startpos+snippetlen]

            if(filetype=="training"):
                d = {'file':str(file),'class':str(file.split('_')[0]),'txt':page_content}
            else:
                d = {'file':str(file),'txt':page_content,'LIBOR_startpage':startpage,'LIBOR_startpageposition':startpagepos,'LIBOR_docposition':startpos,'LIBOR_startsnippet':snippet}#'LIBOR_count':search_word_count,
            
            df = df.append(d,ignore_index=True)
    
    return df


def model_train(df,fileloc):
    df['label'] = df['class']#.map({'mortgage': 0, 'amendment': 1,'normal':2,'term':3,'syndicate':4})
    X_train,y_train = df['txt'], df['label']
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
    X_train_cv = cv.fit_transform(X_train)
    #X_test_cv = cv.transform(X_test)

    word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())
    top_words_df = pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)
    word_freq_df.shape
    word_freq_df.head()
    print(top_words_df.head())

    # Decision tree 2

    # Step 1: Import the model you want to use
    # This was already imported earlier in the notebook so commenting out

    # Step 2: Make an instance of the Model
    clf = DecisionTreeClassifier(max_depth = 7, random_state = 0)
    # Step 3: Train the model on the data
    clf.fit(X_train_cv, y_train)
    # Step 4: Predict labels of unseen (test) data
    # Not doing this step in the tutorial
    # clf.predict(X_test) 
    clf_feature_names = cv.get_feature_names() #list(X_train_cv.columns)
    clf_target_names = [str(s) for s in y_train.unique()]
    
    #fn = cv.get_feature_names()
    #cn = ['mortgage','amendment','normal','term','syndicate']
    #https://mljar.com/blog/visualize-decision-tree/
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf,feature_names=clf_feature_names,class_names=clf_target_names,filled = True)
    fig.savefig(fileloc+'/'+'decistion_tree.png')
    predictions = clf.predict(X_train_cv)
    finaloutput = df.loc[:,['file','class','label']].copy()
    finaloutput['predictions']=predictions
    finaloutput.to_csv(fileloc+'/'+'outputresult.csv',index=False)
    filename = fileloc+'/'+'model.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    filename = fileloc+'/'+'transformer.pkl'
    pickle.dump(cv, open(filename, 'wb'))    

def doc_predict(df):
    loaded_model = pickle.load(open(modelloc+'model.pkl', 'rb'))
    loaded_transformer = pickle.load(open(modelloc+'transformer.pkl', 'rb'))
    X_test = df['txt']
    X_test_cv = loaded_transformer.transform(X_test)
    predictions = loaded_model.predict(X_test_cv)
    finaloutput = df.copy()#df.loc[:,['file']].copy()
    finaloutput['predictions']=predictions
    #finaloutput['class'] = finaloutput['predictions'].map({0:'mortgage', 1:'amendment',2:'normal',3:'term',4:'syndicate'})
    finaloutput.loc[:, finaloutput.columns != 'txt'].to_csv(outputloc+'outputpredict.csv',index=False)
    return finaloutput

def info_extract(text):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Pattern Matching
    from spacy.matcher import Matcher 
    from spacy.tokens import Span 

    # POS pattern for Loan Parties extraction
    link = ''
    entity1 = ''
    entity2 = ''

    for chunk in doc.noun_chunks:
        if chunk.root.head.text.lower() == 'between':
            entity1 = chunk.text
            link = chunk.root.text
        elif (entity1 != '') and (link !='') and (chunk.root.dep_=='conj') :
            entity2 = chunk.text
        elif (chunk.root.head.text == entity2) :
            entity2 = entity2+' '+chunk.text
        elif (chunk.root.head.text == entity1) :
            entity1 = entity2+' '+chunk.text
        elif (entity1 != '') and (link !='') and (entity2 != ''):
            break

    #import re
    #print('Entity 1: ',re.sub(' +', ' ',entity1))
    #print('Entity 2: ',entity2)

    return entity1,entity2
    """
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    from spacy.matcher import PhraseMatcher
 
    def on_match(matcher, doc, id, matches):
      print('Matched!', matches)

    matcher = PhraseMatcher(nlp.vocab)
    matcher.add("LIBOR", on_match, nlp("LIBOR"))
    #matcher.add("HEALTH", on_match, nlp("health care reform"),
    #                              nlp("healthcare reform"))
    #doc = nlp("Barack Obama urges Congress to find courage to defend his healthcare reforms")
    matches = matcher(doc)
    matches
    #for ent in doc.ents:
    #    if()
    #    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    """



if __name__ == "__main__":
    #df = get_input(trainloc,"training")
    #model_train(df,trainloc)
    
    df = get_input(testloc,"testing")
    df = doc_predict(df)
    #print(df['txt'].apply(info_extract))
    df['entity1'],df['entity2'] = zip(*df['txt'].apply(info_extract))
    df.loc[:, df.columns != 'txt'].to_csv(outputloc+'output.csv',index=False)
    pass


