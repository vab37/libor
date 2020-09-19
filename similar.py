fileloc = "./similarity/"
from sklearn.feature_extraction.text import TfidfVectorizer
import os
#for file in os.listdir(fileloc):
filelist = os.listdir(fileloc)
#print(filelist)
documents = [open(fileloc+f).read() for f in filelist]
#print(os.listdir(fileloc))
tfidf = TfidfVectorizer().fit_transform(documents)
# no need to normalize, since Vectorizer will return normalized tf-idf
pairwise_similarity = tfidf * tfidf.T
#print(pairwise_similarity.toarray())

import numpy as np     
arr = pairwise_similarity.toarray()     
np.fill_diagonal(arr, np.nan)                                                                                                                                                                                                                            
#print(arr)
input_doc = "inputtext.txt"                                                                                                                                                                                                  
input_idx = filelist.index(input_doc)                                                                                                                                                                                                                      
#print('inputindex:',input_idx)                                                                                                                                                                                                                                                
result_idx = np.nanargmax(arr[input_idx])  
#print(result_idx) 
print('Recommended document which matches inputtext.txt:',filelist[result_idx])
