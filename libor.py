
import PyPDF2 
import slate3k as slate
import pandas as pd
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import re
trainloc = "./train/"
testloc = "./test/"
outputloc = "./output/"
modelloc = "./model/"


def get_input(fileloc,filetype):
    df = pd.DataFrame()
    for file in os.listdir(fileloc):
        #print(file)
        if file.endswith(".pdf"):
            #pdfFileObj = open(fileloc+"/"+file, 'rb') 
            #pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
            #print(pdfReader.numPages) 
            with open(fileloc+"/"+file, 'rb') as f:
                pages = slate.PDF(f)
            print(len)
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



            if(filetype=="training"):
                d = {'file':str(file),'class':str(file.split('_')[0]),'txt':page_content}
            else:
                d = {'file':str(file),'txt':page_content,'LIBOR_startpage':startpage,'LIBOR_startpageposition':startpagepos,'LIBOR_docposition':startpos,'LIBOR_startsnippet':snippet,'fallbackPresent':hasfallback,'fallbackPosition':startpos2,'fallbackPage':startpage2,'fallbackText':snippet2,'fallbackTextComplexity':0 }
            
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
        elif (entity1 != '') and (entity2 != ''): #and (link !='') 
            break

    #import re
    #print('Entity 1: ',re.sub(' +', ' ',entity1))
    #print('Entity 2: ',entity2)
    from spacy.matcher import Matcher
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": "effective"}, {"LOWER": "date"},{"IS_PUNCT": True},{"ENT_TYPE":"DATE","OP":"*"}]#{"ENT_TYPE": "DATE","OP":"*"}]
    matcher.add("Effective Date", None, pattern)
    matches = matcher(doc)
    matchtext = ""
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        matchtext = span.text
        #print(match_id, string_id, start, end, span.text)
    #print('matchtext:',matchtext)
    subdoc = nlp(matchtext)
    effective_date = ''
    for ent in subdoc.ents:
        if(ent.label_ == 'DATE'): 
            effective_date = ent.text
    #print(effective_date)
    
    ########################
    
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": "termination"}, {"LOWER": "date"},{"IS_PUNCT": True},{"ENT_TYPE":"DATE","OP":"+"}]#{"ENT_TYPE": "DATE","OP":"*"}]
    matcher.add("Termination Date", None, pattern)
    matches = matcher(doc)
    matchtext = ""
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        matchtext = span.text
        #print(match_id, string_id, start, end, span.text)
    #print('matchtext:',matchtext)
    subdoc = nlp(matchtext)
    termination_date = ''
    for ent in subdoc.ents:
        if(ent.label_ == 'DATE'): 
            termination_date = ent.text
    #print(termination_date)

    ##########################################

    doc = nlp(text.replace(" USD ", " $ "))
    #for ent in doc.ents: 
        #print(ent.text, ent.start_char, ent.end_char, ent.label_)

    #for tok in doc: 
        #print("token start<",tok.text,">token end")#, "-->",tok.dep_,"-->", tok.pos_)

    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": "notional"}, {"LOWER": "amount"},{"IS_PUNCT": True},{"TEXT": '$'},{"ENT_TYPE":"MONEY","OP":"*"}]
    matcher.add("Notional Amount", None, pattern)
    matches = matcher(doc)
    matchtext = ""
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        matchtext = span.text
        #print(match_id, string_id, start, end, span.text)
    #print('matchtext:',matchtext)
    subdoc = nlp(matchtext)
    notional_amt = ''
    for ent in subdoc.ents:
        if(ent.label_ == 'MONEY'): 
            notional_amt = ent.text
    #print(notional_amt)

    return entity1,entity2,effective_date,termination_date,'USD',notional_amt




if __name__ == "__main__":
    #df = get_input(trainloc,"training")
    #model_train(df,trainloc)
    
    df = get_input(testloc,"testing")
    df = doc_predict(df)
    #print(df['txt'].apply(info_extract))
    df['entity1'],df['entity2'],df['startdate'],df['terminationdate'],df['currency'],df['amount'] = zip(*df['txt'].apply(info_extract))
    df.loc[:, df.columns != 'txt'].to_csv(outputloc+'output.csv',index=False)
    pass


