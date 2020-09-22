import sys

#print ('Number of arguments:', len(sys.argv), 'arguments.')
#print ('Argument List:', str(sys.argv))

text1 = sys.argv[1]

#Interest duration will be 2 months if the Borrowing does not specify specify Interest duration of LIBOR Advances. Bank will rely on information provide in Invoice Transmittal, Borrowing Base Certificate and Notice of Borrowing. Borrower will be responsible for any loss suffered by the Bank suffered by the bank due to this.

text2 = sys.argv[2]
#the duration of the Interest Period applicable to any such LIBOR Advances included in such notice; provided that if the Notice of Borrowing shall fail to specify the duration of the Interest Period for any Advance comprised of LIBOR Advances, such Interest Period shall be one (1) month. Bank may rely on information set forth in or provided with the Invoice Transmittal, Borrowing Base Certificate, Purchase Order Transmittal and Notice of Borrowing. Borrower will indemnify Bank for any loss Bank suffers due to such reliance.

from sklearn.feature_extraction.text import TfidfVectorizer
documents = [text1,text2]
#print(os.listdir(fileloc))
tfidf = TfidfVectorizer().fit_transform(documents)
# no need to normalize, since Vectorizer will return normalized tf-idf
pairwise_similarity = tfidf * tfidf.T
#print(pairwise_similarity.toarray())

import numpy as np     
arr = pairwise_similarity.toarray()     
np.fill_diagonal(arr, np.nan)                                                                                                                                                                                                                            
print(arr[0,1])
#input_doc = "inputtext.txt"                                                                                                                                                                                                  
#input_idx = filelist.index(input_doc)                                                                                                                                                                                                                      
#print('inputindex:',input_idx)                                                                                                                                                                                                                                                
#result_idx = np.nanargmax(arr[input_idx])  
#print(result_idx) 
#print('Recommended document which matches inputtext.txt:',filelist[result_idx])
