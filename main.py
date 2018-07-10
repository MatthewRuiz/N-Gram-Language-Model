# Used to get input from command line

import argparse
from nltk.util import ngrams

from read_csv import Read_CSV as r
from unigrams import Unigrams as u
from bigrams import Bigrams as b
from trigrams import Trigrams as t
import time
import re
from prettytable import PrettyTable

uni = u()
bi = b()
tri = t()
nd = None
START_SENTENCE_BOUNDARY_MARKER = '<s>'


def main(argv):
    csv_file = argv.file

    if csv_file is None:
        csv_file = 'Reviews.csv'

    read_csv = r(csv_file)

    start_time = time.time()

    print('Please wait while the unigrams, bigrams, and trigrams are '
          + 'generated for the file: ' + csv_file)
    size_of_dataset = 0
    lines = 0
    # For the same amount of lines,
    # N of 'amazon-reviews.csv' is roughly 2N of 'winemag-data_first150k.csv
    # in the amount of data, we account for the difference by changing the
    # amount of text.
    if argv.lines is None :
        lines = 50000 if argv.file == 'Reviews.csv' else 100000
    else:
        lines = argv.lines

    for row in read_csv.load_csv(lines):
        row = re.sub(r'\.{2,}<br />|<br>|<[a-z].*>|</a>', '.', row) # Map HTML tags and multuple occurences of periods to '.'
        row = re.sub(r'[@#%^&*()(_)-+=";:]', '$', row)               # Map several special symbols to $
        row = re.sub(r'\d*\.*\d+', '0.0', row)                       # Map all digits to 0.0
        tokenized_string_as_list = re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", row)   # Tokenize remaining text

        size_of_dataset += len(tokenized_string_as_list)
        generate_ngrams(tokenized_string_as_list)

    print()

    # Necessary computation
    uni.set_unigram_count_total()
    uni.set_unigram_probabilities()
    bi.set_bigram_frequency_counts()
    bi.set_bigram_probabilities()
    tri.set_trigram_frequency_counts()
    tri.set_trigram_probabilities()

    print('The first {} lines in the {} file contained {} words.'.format(lines,
                                                                     argv.file,
                                                                     size_of_dataset))
    print('It took {:.2f} seconds to generate the unigram, bigram, and trigram models for {}'
            .format(time.time() - start_time, argv.file ))
    menu()


def menu():
    """Program menu"""
    print()
    print('N-gram Language Models')
    print()
    ngram_menu_selections = ['Unigram', 'Bigram', 'Trigram', 'Quit'] #  'All',

    for x in range(0, len(ngram_menu_selections)):
        print(str((x+1)) + ') ' + ngram_menu_selections[x])

    ngram_selection = ''
    while True:
        ngram_selection = input('Please select the ngram you would like to work on: ')

        if(not ngram_selection.isdigit()     # Not a digit.
                or 1 > int(ngram_selection)  # Digit too low.
                or int(ngram_selection) > len(ngram_menu_selections) # Digit too high.
        ):
            print('Please enter a digit (1-{})'.format(str(len(ngram_menu_selections))))
            continue
        else:
            break

    print()
    ngram_menu(ngram_selection)


def ngram_menu(selection):

    switch = {
            '1': unigram_menu,
            '2': bigram_menu,
            '3': trigram_menu,
            # '4': all_ngram_menu,
            '4': get_quit
            }

    func = switch.get(selection, lambda: 'Nothing')
    return func()


def unigram_menu():
    """
        Unigram menu

        Handle flow of operation for unigram-specific operations.
    """
    unigram_menu = ['Sorted Unigrams', 'Find count of unigram',
                    'Find probability of unigram', 'Generate random sentence',
                    'Choose different N-gram', 'Quit application']

    return_to_menu = False

    while True:
        print('PLEASE SELECT ONE OF THE FOLLOWING OPERATIONS')

        for x in range(0, len(unigram_menu)):
            print(str((x+1)) + ') ' + unigram_menu[x])
        print()

        menu_selection = input('Please select an operation(1-{}): '
                               .format(len(unigram_menu)))

        if menu_selection == '1':   # Display sorted unigrams
            k = input('How many unigrams would you like to see?')
            while not k.isdigit():
                k = input('Please enter a digit: ')
            uni.print_unigrams(k=int(k))
            print()
        elif menu_selection == '2': # Display count of unigram
            unigram = input('Please enter the unigram: ').strip()
            print()
            count = uni.get_unigram_count(unigram)
            if count != -1: # Unigram was found
                print('Unigram Counts')
                table = PrettyTable(['Wn', 'C(Wn)']) # PrettyTable headers
                table.add_row([unigram, count])      # Populate table
                print(table)
                print()
        elif menu_selection == '3': # Display probability of unigram
            unigram = input('Please enter the unigram: ').strip()
            print()
            prob = uni.get_unigram_probability(unigram)
            if prob != -1: # Unigram was found
                prob = '{0:.6f}'.format(prob)
                print('Unigram Probability')
                table = PrettyTable(['Wn', 'P(W)'])
                table.add_row([unigram, prob])
                print(table)
                print()
        elif menu_selection == '4':
            uni.generate_random_sentence()
        elif menu_selection == '5': # Return to n-gram selection menu
            return_to_menu = True
            break
        elif menu_selection == '6': # Quit application
            get_quit()
        else:
            print('You did not enter a valid menu choice.')
            return_to_menu = True

    if return_to_menu:
        menu()

def bigram_menu():

    bigram_menu = ['Sort by count', 'Find bigrams from Wn-1',
                    'Find count of bigram', 'Find probability of bigram',
                    'Generate random sentence', 'Choose different N-gram',
                    'Quit application']
    return_to_menu = False

    while True:
        print('PLEASE SELECT ONE OF THE FOLLOWING OPERATIONS')

        for x in range(0, len(bigram_menu)):
            print(str((x+1)) + ') ' + bigram_menu[x])
        print()

        menu_selection = input('Please select an operation(1-{}): '
                               .format(len(bigram_menu)))

        if menu_selection == '1': # Display sorted bigrams by count
            k = input('How many bigrams would you like to see?')
            while not k.isdigit():
                k = input('Please enter a digit: ')
            bi.print_bigrams(k=int(k))
            print()
        elif menu_selection == '2': # Find bigrams with a similar Wn-1
            wn_minus_one = input('Please enter Wn-1: ').strip()
            k = input('Please enter a number k, to output the top k with the highest count: ')
            while not k.isdigit():
                k = input('Please enter a digit: ')
            bi.get_bigrams_for_given_wn_minus_one(wn_minus_one, int(k))
            print()
        elif menu_selection == '3': # Display count of bigram
            wn_minus_one = input('Please enter Wn-1: ').strip()
            wn = input('Please enter Wn: ').strip()
            bigram = (wn_minus_one, wn)
            print()
            count = bi.get_bigram_count(bigram)
            if count != -1: # Bigram found
                print('Bigram Count')
                table = PrettyTable(['(Wn-1, Wn)', 'C(W)'])
                table.add_row([bigram, count])
                print(table)
        elif menu_selection == '4': # Display probability of bigram
            wn_minus_one = input('Please enter Wn-1: ').strip()
            wn = input('Please enter Wn: ').strip()
            bigram = (wn_minus_one, wn)
            print()
            prob = bi.get_bigram_probability(bigram)
            if prob != -1: # Bigram found
                prob = '{0:.6f}'.format(prob)
                print('Bigram Probability')
                table = PrettyTable(['(Wn-1, Wn)', 'P(W)'])
                table.add_row([bigram, prob])
                print(table)
        elif menu_selection == '5':
            bi.generate_random_sentence()
        elif menu_selection == '6':
            return_to_menu = True
            break
        elif menu_selection == '7':
            get_quit()

    if return_to_menu:
        menu()

def trigram_menu():

    trigram_menu = ['Sort by count', 'Find trigrams from Wn-2 and Wn-1',
                    'Find count of trigram', 'Find probability of trigram',
                    'Generate random sentence', 'Choose different N-gram',
                    'Quit application']
    return_to_menu = False

    while True:
        print('PLEASE SELECT ONE OF THE FOLLOWING OPERATIONS')

        for x in range(0, len(trigram_menu)):
            print(str((x+1)) + ') ' + trigram_menu[x])
        print()

        menu_selection = input('Please select an operation(1-{}): '
                               .format(len(trigram_menu)))

        if menu_selection == '1':
            k = input('How many trigrams would you like to see?')
            while not k.isdigit():
                k = input('Please enter a digit: ')
            tri.print_trigrams(k=int(k))
            print()
        elif menu_selection == '2': # Display trigrams with similar (Wn-2, Wn-1)
            print()
            wn_minus_two = input('Please enter Wn-2: ').strip()
            wn_minus_one = input('Please enter a value for Wn-1: ').strip()
            prev_bigram = (wn_minus_two, wn_minus_one)
            k = input('Please enter a number k, to output the top k with the highest count: ')
            while not k.isdigit():
                k = input('Please enter a digit: ')
            tri.get_trigrams_for_given_pre_bigram(prev_bigram, int(k))
            print()
        elif menu_selection == '3':
            wn_minus_two = input('Please enter Wn-2: ').strip()
            wn_minus_one = input('Please enter Wn-1: ').strip()
            wn = input('Please enter Wn: ').strip()
            trigram = (wn_minus_two, wn_minus_one, wn)
            print()
            count = tri.get_trigram_count((trigram))
            if count != -1:
                table = PrettyTable(['(Wn-2, Wn-1, Wn)', 'C(W)'])
                table.add_row([trigram, count])
                print(table)
        elif menu_selection == '4':
            wn_minus_two = input('Please enter Wn-2: ').strip()
            wn_minus_one = input('Please enter Wn-1: ').strip()
            wn = input('Please enter Wn: ').strip()
            trigram = (wn_minus_two, wn_minus_one, wn)
            print()
            prob = tri.get_trigram_probability((trigram))
            if prob != -1:
                prob = '{0:.6f}'.format(prob)
                print('Trigram Probability')
                table = PrettyTable(['(Wn-2, Wn-1, Wn)', 'P(W)'])
                table.add_row([trigram, prob])
                print(table)
        elif menu_selection == '5':
            tri.generate_random_sentence()
        elif menu_selection == '6':
            return_to_menu = True
            break
        elif menu_selection == '7':
            get_quit()

    if return_to_menu:
        menu()

def generate_ngrams(text):
    """
    Generate unigrams, bigrams and trigrams. Ngrams are created simultaneously
    to prevent unecessary looping.

    :param text: The text to be processed

    """
    start_of_sentence = True # Used to generate ngrams at the beginning of a sentence.

    for x in range(len(text)):
        token = text[x].lower()
        token_plus_one = text[x+1].lower() if x < len(text) - 1 else None
        token_plus_two = text[x+2].lower() if x < len(text) - 2 else None

        if token is '.':
            start_of_sentence = True
        uni.generate_unigram(token)
        bi.generate_bigram(token, token_plus_one, start_of_sentence)
        tri.generate_trigram(token, token_plus_one, token_plus_two, start_of_sentence)
        if start_of_sentence is True:
            start_of_sentence = False

def get_quit():
    """Program menu"""
    wants_to_quit = input('Do you want to quit(Y/N)? ')
    if wants_to_quit[0].lower() == 'y':
        quit()
    else:
        menu()

def parse_arguments():
    # Holds all the information necessary to aprse the command line into Python data types.
    parser = argparse.ArgumentParser()

    # optional arguments:
    #   -h              Show this help message and exit.
    #   -f, --file      Is this run a test run?
    #   -l, --lines     The amount of lines in --file that are processed.
    #   -r, --row


    parser.add_argument('-f', '--file',
                        help='The name of csv file in the format of filename.csv',
                        type=str)
    parser.add_argument('-l', '--lines',
                        help='The amount of lines of text to be processed.',
                        type=int)
    parser.add_argument('-c', '--column',
                        help="""If the user wishes to use their own .csv file, the index of
                                    the column that holds the text should be entered.
                                    For example: the text for the winemag-data_first150k.csv .csv file
                                    is in the 3rd column. Therefore, a 2 would be entered. """
                        )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
