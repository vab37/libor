# libor
## Sample commands
Install Git Bash\
Change to folder: cd C:/Users/agvai/OneDrive/Documents/out/code/libor \
Create enironment: conda env create -f environment.yml \
Activate environment: source activate liborenv \
Run code: python libor.py 

Other commands\
python -m spacy download en_core_web_sm

## Similarity algo working
Run similar.py using a commandline instruction below (2 text strings as commandline argument) and it returns the similarity between the strings on commandline \
python similar.py "Interest duration will be 2 months if the Borrowing does not specify specify Interest duration of LIBOR Advances. Bank will rely on information provide in Invoice Transmittal, Borrowing Base Certificate and Notice of Borrowing. Borrower will be responsible for any loss suffered by the Bank suffered by the bank due to this." "the duration of the Interest Period applicable to any such LIBOR Advances included in such notice; provided that if the Notice of Borrowing shall fail to specify the duration of the Interest Period for any Advance comprised of LIBOR Advances, such Interest Period shall be one (1) month. Bank may rely on information set forth in or provided with the Invoice Transmittal, Borrowing Base Certificate, Purchase Order Transmittal and Notice of Borrowing. Borrower will indemnify Bank for any loss Bank suffers due to such reliance."

## Classify.py algo working
Sample command takes location of the file to be classified and parameters to suppress any warnings on commandline. Outputs the predicted class of the document: \
python -W ignore classify.py './test/irswap_DEUTSCHE BANK_LIBOR_TEXT_edit.pdf' 

Also make sure model folder contains the required model
