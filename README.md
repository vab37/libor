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
Similarity folder contains the following files:
recommendation1-4.txt: these are text we would like to recommend. File names can be changed to any name and any number of such documents can be added \ 
inputtext.txt: This is the text which will be matched to the recommended text. The file name should be kept same
Run similar.py. The program displays the recommended document which matches the inputtext.txt

