INTRODUCTION
------------

This program provides a set of natural language tools written in Python.
It can take a .csv file, find a specific column, provided by the user should they
chose to use a new .csv file, and generate unigram, bigram and trigram models. Text was
normalized to fit model. See source documentation for the explanation. Once generated, 
you may retrieve the observed frequency and probability for a given n-gram; as well
as other interesting commands. Ultimately, all of the tools are used to randomly generate
sentences using the different n-gram models.

REQUIREMENTS
------------

This program requires the following packages:

	* argparse (https://pypi.python.org/pypi/argparse)
	* NLTK (https://www.nltk.org/data.html)
	* PrettyTable (https://pypi.python.org/pypi/PrettyTable)
	
This program requires the following .csv files:

	* Amazone Fine Food Reviews
		(https://www.kaggle.com/snap/amazon-fine-food-reviews/data)
	* Wine Reviews
		(https://www.kaggle.com/zynicide/wine-reviews/data)
		
**Changing the file name will result in a FileNotFoundError**
	
BUILD INSTRUCTIONS
------------------

There are three optional parameters:

    *   -f FILE, --file FILE
                The name of csv file in the format of filename.csv
    *   -l LINES, --lines LINES
                The amount of lines of text to be processed.
    *   -c COLUMN, --column COLUMN
                If the user wishes to use their own .csv file, the
                index of the column that holds the text should be
                entered. For example: the text for the winemag-
                data_first150k.csv .csv file is in the 3rd column.
                Therefore, a 2 would be entered.
           
Should no parameters be given, the following default paremeters will be given:

    * -f Reviews.csv
    * -l 50000
    * -c 9

If you are using the terminal, navigate to the directory containing the _main.py_ file.

   1) Entering the command: `python main.py -h` will provide you with the
    above instructions.

If you are using an IDE, you can also run the *main.py* file with or without providing 
parameters.




		
	