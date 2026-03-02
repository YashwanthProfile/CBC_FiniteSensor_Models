# To check the directory of python interpreter

``` which python3 ```

# To change the pthon interpreter to local project dir

## Navigate to the project dir in the terminal 
``` cd to_project_dir ```

## Verify the current working dir
``` pwd ```

## Create a venv
``` python3 -m venv venv ```

## Verify by checking if a venv file is created in pwd

## Activate the venv (everytime we reopen the terminal) - this should change the python interpreter dir
``` source ./venv/bin/activate```
    
    - In VS-Code to set this as a default python interpreter dir (atleast unitll the project-dir is not removed from WORKSAPCE (on left panel))
    - Press: Ctrl + Shift + P 
    - > Select Interpreter
    - > Choose the venv from the current project dir
    - This will temporarily set the desired dir for python interpreter. Even after closing the terminal the VS code automatically activates venv
    - When the dir is removed from WORKSAPCE navigation panel the python interpreter default to global 

## Verify that the current python interpreter dir venv from current project
``` which python3 ``` 

# To install required lib
``` pip install -r requirements.txt ```

## To create requirement file: Note that this file will contain libraries which were not explicitly install (those are essentail)
``` pip freeze > requirements.txt ```

## Verify installation of required lib: This prints out every dependent lib installed for the main lib too
``` pip list ```

## Verify only the main lib requirement
``` pip list --not-required ```