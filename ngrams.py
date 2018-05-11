from prettytable import PrettyTable
import random
import collections
import math

class Ngrams():

    # initialize
    def __init__(self, n):
        self.N = n
        self.V = 0
        self.OBSERVED_FREQUENCIES = {}
        self.PROBABILITIES = {}
        # Equivalent to length of tokens inside of text
        self.FREQUENCY_COUNTS = {}
        # Intervals for each unigram probability. Formatted as {N-gram : (low, high)}
        self.PROB_INTERVALS = {}

        # Prevent unnecessary sorting
        self.ngrams_are_sorted = False
        self.ngram_intervals_are_recorded = False

    def set_ngram_probabilities(self):

        if(self.ngrams_are_sorted is False):
            self.sort_ngram_by_observed_frequency()

        ngram_probabilities = collections.OrderedDict()

        for ngram in self.OBSERVED_FREQUENCIES:
            prefix = self.get_prefix(ngram)
            observed_frequency = self.get_ngram_observed_frequency(ngram)
            prob = float(observed_frequency / self.FREQUENCY_COUNTS[prefix])
            ngram_probabilities[ngram] = prob

        self.PROBABILITIES = ngram_probabilities

    def set_ngram_frequency_counts(self):

        bigram_freq_counts = {}
        for x in self.OBSERVED_FREQUENCIES:
            prefix = self.get_prefix(x)
            observed_frequency = self.get_ngram_observed_frequency(x)
            if len(prefix) == 1:
                prefix = prefix[0]

            if (prefix not in b):
                bigram_freq_counts[prefix] = observed_frequency
            else:
                bigram_freq_counts[prefix] += observed_frequency

        self.FREQUENCY_COUNTS = bigram_freq_counts

    def set_ngram(self, ngram):
        """
        Add ngram to OBSERVED_FREQUENCIES dict if it does not already exists;
        otherwise, increment count.

        :param ngram: ngram to insert or increment
        """
        if (ngram not in self.OBSERVED_FREQUENCIES.keys()):
            self.OBSERVED_FREQUENCIES[ngram] = 1
        else:
            self.OBSERVED_FREQUENCIES[ngram] += 1

    def get_ngram_observed_frequency(self, ngram):
        """
        Return the count for a specific ngram.

        :param ngram: ngram to search for
        :rtype: int
        """
        try:
            return self.OBSERVED_FREQUENCIES[ngram]
        except KeyError:
            print('The ngram you have entered, {} , could not be found\n'
                   .format(ngram))
            return -1

    def get_ngram_probability(self, ngram):
        """
        Return the probability for a ngram.

        :param ngram: ngram to search for

        :rtype: float
        """
        try:
            return self.PROBABILITIES[ngram]
        except KeyError:
            print('The ngram you have entered, {} , could not be found\n'
                   .format(ngram))
            return -1

    def get_ngram_prob_intervals(self, ngram):
        """
        Get the probability interval for ngram with the same prefix.

        :param ngram: used to find the ngrams of interest

        :rtype: dict
        """
        prefix = self.get_prefix(ngram)

        try:
            return self.PROB_INTERVALS[prefix]
        except KeyError:
            print('The prefix you have entered, {} , could not be found\n'
                   .format(prefix))
            return -1

    def get_prefix(self, ngram):
        prefix = ()

        if len(ngram) == 1:
            prefix = (ngram)
        else:
            for counter, wn in enumerate(ngram):
                if counter < len(ngram) - 1:
                    prefix += (wn,)

        return prefix

    def compute_sentence_probability(self, sentence_tokens):

        sentence_log_probability = 0.0
        for ngram in sentence_tokens:
            ngram_log_probability = 0
            ngram_prob = self.get_trigram_probability(ngram)
            if(ngram_prob != -1):
                ngram_log_probability = math.log(ngram_prob)

            # (P1*P2*P3*P4) == exp(log(P1)+log(P2)+log(P3)+log(P4))
            sentence_log_probability += ngram_log_probability

        print('The probability of generating the sentence above in log scale is: {0:.20f}'.format(sentence_log_probability))

    def sort_ngram_by_observed_frequency(self):
        """Sort ngrams by observed frequency"""
        sorted_ngrams = sorted(self.OBSERVED_FREQUENCIES,
                               key=lambda t: self.OBSERVED_FREQUENCIES[t],
                               reverse=True)

        ordered_dict_of_ngram_observed_frequencies = collections.OrderedDict()

        # Store the sorted ngrams inside of the
        # OrderedDict: ordered_dict_of_ngram_observed_frequencies
        for ngram in sorted_ngrams:
            observed_frequency = self.get_ngram_observed_frequency(ngram)
            ordered_dict_of_ngram_observed_frequencies[ngram] = observed_frequency

        self.OBSERVED_FREQUENCIES = ordered_dict_of_ngram_observed_frequencies
        self.ngrams_are_sorted = True

    def is_punctuation(self, token):
        """
        Return True if the token is a punctuation.

        :param token: wn(current word)
        :rtype: boolean
        """
        return token in['.', '!', '?']

    def set_ngram_prob_intervals(self, ngram, prob_intervals):
        prefix = self.get_prefix(ngram)
        if(prefix not in self.PROB_INTERVALS.keys()):
            self.PROB_INTERVALS[prefix] = prob_intervals

    def generate_start_of_sentence_ngram(self, token, ngram_length):
        ngram = []
        for x in range(0, ngram_length - 1):
            ngram.append('<s>')

        ngram.append(token)
        self.set_ngram(tuple(ngram))

    def generate_ngram(self, tokens, start_of_sentence):
        ngram = []
        tokens_length = len(tokens)

        for x in range(0, tokens_length):
            token = tokens[x]
            if x == (tokens_length-1):
                if self.is_punctuation(token):
                    token = '</s>'
            elif x == 0:
                if self.is_punctuation(token):
                    token = '<s>'
            else:
                if token is None or self.is_punctuation(token):
                    return

            ngram.append(token)

            if start_of_sentence:
                if self.is_punctuation(tokens[x]):
                    try:
                        self.generate_start_of_sentence_ngram(tokens[x+1], tokens_length)
                    except IndexError:
                        return
                else:
                    self.generate_start_of_sentence_ngram(tokens[x], tokens_length)

                start_of_sentence = False

        if None not in ngram:
            self.set_ngram(tuple(ngram))

    def print_ngrams(self, k):
        top = k

        if self.ngrams_are_sorted is False:
            self.sort_ngram_by_observed_frequency()

        print()
        print('Ngram Count: V = {}'.format(len(self.OBSERVED_FREQUENCIES)))
        column_names = self.create_ngram_column_headers(self.N)
        table = PrettyTable(column_names)

    def create_ngram_column_headers(self, n, probability=False):
        column_names = []

        for x in reversed(range(n)):
            if x is not 0:
                formatted_wn = 'Wn-{}'.format(x)
            else:
                formatted_wn = 'Wn'

            column_names.append(formatted_wn)

        last_column_header = self.create_last_column_header(n, probability)
        column_names.append(last_column_header)

        return column_names

    def create_last_column_header(self, n, probability):
        last_column_header = ''

        for x in reversed(range(n)):
            if x is not 0:
                last_column_header += 'Wn-{}'.format(x)
            else:
                last_column_header = 'Wn|' + last_column_header

        if probability:
            return 'P({})'.format(last_column_header)
        else:
            return 'C({})'.format(last_column_header)

