import csv

class Read_CSV():

    # initialize
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.column_with_text = 9 if self.csv_file == 'Reviews.csv' else 2

    def load_csv(self, lines):
        """
            Read and return each row of .csv file individually.

        :param lines: The amount of lines to be processed
        :return: yield - single row of data
        """
        text = ""
        with open(r'{}'.format(self.csv_file),
                                encoding='ascii',
                                errors='ignore') as csv_data_file:

            csv_reader = csv.reader(csv_data_file)
            count = 0
            for row in csv_reader:
                if lines is not -1: # Check if user has entered a number for lines.
                    if(count == 0):
                        count += 1
                    elif(count < lines):
                        yield row[self.column_with_text].lower()
                        count += 1
                    else:
                        return text
                else:  # Default: read all lines
                    if(count == 0):
                        count += 1
                    else:
                        yield row[self.column_with_text].lower()
