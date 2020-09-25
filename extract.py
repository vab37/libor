
import pandas as pd
import os
import re
import sys

#print ('Number of arguments:', len(sys.argv), 'arguments.')
#print ('Argument List:', str(sys.argv))

filename = sys.argv[1]
#filename = './test/irswap_DEUTSCHE BANK_LIBOR_TEXT_edit.pdf'
def get_input(filename):
    df = pd.DataFrame()
    f = open(filename, "r", encoding="utf8")
    d = {'file':str(filename),'txt':f.read() }
    df = df.append(d,ignore_index=True)
    return df


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
    df = get_input(filename)
    df.head()
    #df = pd.DataFrame()
    #df.loc[0,'txt']
    df['entity1'],df['entity2'],df['startdate'],df['terminationdate'],df['currency'],df['amount'] = info_extract(df.loc[0,'txt']) #zip(*df['txt'].apply(info_extract))
    df.loc[0,['entity1','entity2','startdate','terminationdate','currency','amount']]
    print(df.loc[0,['entity1','entity2','startdate','terminationdate','currency','amount']].to_json())
    #df.loc[i].to_json("row{}.json".format(i))
    #df.loc[:, df.columns != 'txt'].to_csv(outputloc+'output.csv',index=False)
    pass


