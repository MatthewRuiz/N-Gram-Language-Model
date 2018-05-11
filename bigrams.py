from prettytable import PrettyTable
import random
import collections
import math

class Bigrams():

    # initialize
    def __init__(self):
        self.BIGRAMS = {}
        self.BIGRAM_PROBABILITIES = {}
        # The frequency counts for each bigram prefix
        self.FREQUENCY_COUNT = {}
        # Intervals for each bigram probability. Formatted as {(wn_minus_one, Wn) : (low, high)}
        self.BIGRAM_PROB_INTERVALS = {}

        self.bigrams_are_sorted = False

    def set_bigram(self, bigram):
        """
        Add bigram to BIGRAMS dict if it does not already exists;
        otherwise, increment count.

        :param bigram: bigram to insert or increment
        """
        if (bigram not in self.BIGRAMS.keys()):
            self.BIGRAMS[bigram] = 1
        else:
            self.BIGRAMS[bigram] += 1

    def set_bigram_probabilities(self):
        """Compute and record the probabilities for every bigram."""
        if self.bigrams_are_sorted is False:
            self.sort_bigram_by_count()

        bigram_probabilities = collections.OrderedDict()
        for bigram_and_count in self.BIGRAMS.items():
            bigram = bigram_and_count[0]
            wn_minus_one = bigram[0]
            bigram_count = bigram_and_count[1]
            prob = float(bigram_count / self.FREQUENCY_COUNT[wn_minus_one])
            bigram_probabilities[bigram] = prob

        self.BIGRAM_PROBABILITIES = bigram_probabilities

    def set_bigram_frequency_counts(self):
        """Compute and return the counts for each bigram prefix"""
        bigram_freq_counts = {}

        for x in self.BIGRAMS.items():
            bigram = x[0]
            wn_minus_one = bigram[0]
            observed_frequency = x[1]
            if (wn_minus_one not in bigram_freq_counts):
                bigram_freq_counts[wn_minus_one] = observed_frequency
            else:
                bigram_freq_counts[wn_minus_one] += observed_frequency

        self.FREQUENCY_COUNT = bigram_freq_counts

    def set_bigram_prob_intervals(self, wn_minus_one, prob_intervals):
        """
        Set probability interval for bigram if it does not exist.

        :param wn_minus_one: the base for the bigrams of interest -- will act as the key
        :param prob_intervals: dictionary containing the (bigram, interval) values for
            bigrams with the same wn_minus_one -- will act as value
        """
        if wn_minus_one not in self.BIGRAM_PROB_INTERVALS.keys():
            self.BIGRAM_PROB_INTERVALS[wn_minus_one] = prob_intervals

    def get_bigram_count(self, bigram):
        """
        Return the count for a specific bigram.

        :param bigram: bigram to search for
        :rtype: int
        """
        try:
            return self.BIGRAMS[bigram]
        except KeyError:
            print('The bigram you have entered, {} , could not be found\n'
                   .format(bigram))
            return -1

    def get_bigram_probability(self, bigram):
        """
        Return the probability for a bigram.

        :param bigram: bigram to search for
        :param bigram_probabilities: dictionary containing bigram probabilities

        :rtype: float
        """
        try:
            return self.BIGRAM_PROBABILITIES[bigram]
        except KeyError:
            print('The bigram you have entered, {} , could not be found\n'
                   .format(bigram))
            return -1

    def get_random_bigram(self, prev_word):
        """
        Return a random bigram

        :param prev_word: the base for the bigrams of interest

        :rtype: tuple
        """

        # Prevent duplicate sorting
        if prev_word not in self.BIGRAM_PROB_INTERVALS.keys():
            self.generate_bigram_prob_intervals(prev_word)

        # Dictionary containing the intervals for bigrams with the same value for wn_minus_one
        prev_word_bigram_intervals = self.get_bigram_prob_intervals(prev_word)

        rand_num = random.uniform(0.0, 1.0)
        rand_bigram = ''

        # Find the bigram where rand_num falls between its interval
        for bigram in prev_word_bigram_intervals:
            interval = prev_word_bigram_intervals[bigram]
            low = interval[0]  # lower boundary
            high = interval[1]  # upper boundary

            if low < rand_num <= high:  # random word is found
                rand_bigram = bigram
                break

        return rand_bigram

    def get_bigram_prob_intervals(self, wn_minus_one):
        """
        Get the probability interval for bigrams with the same wn_minus_one

        :param wn_minus_one: the base for the bigrams of interest

        :rtype: dict
        """
        try:
            return self.BIGRAM_PROB_INTERVALS[wn_minus_one]
        except KeyError:
            print('The bigram you have entered, {} , could not be found\n'
                   .format(wn_minus_one))
            return -1

    def get_bigrams_for_given_wn_minus_one(self, wn_minus_one, k):
        # Prevent unnecessary sorting.
        if wn_minus_one not in self.BIGRAM_PROB_INTERVALS.keys():
            self.generate_bigram_prob_intervals(wn_minus_one)

        if self.get_bigram_prob_intervals(wn_minus_one) == -1:
            return
        table = PrettyTable(['(Wn-1, Wn)', 'C(Wn-1, Wn)', 'P(Wn-1, Wn)'])

        # Store bigrams
        dict_of_bigram_and_counts = {}

        for bigram in self.BIGRAM_PROB_INTERVALS[wn_minus_one]:
            dict_of_bigram_and_counts[bigram] = self.get_bigram_count(bigram)

        bigrams_sorted_by_count = sorted(dict_of_bigram_and_counts, key=lambda x: dict_of_bigram_and_counts[x], reverse=True)

        for counter, bigram in enumerate(bigrams_sorted_by_count):
            if counter < k:
                bigram_count = self.get_bigram_count(bigram)
                prob = self.get_bigram_probability(bigram)
                prob = '{:.6f}'.format(prob)
                table.add_row([bigram, bigram_count, prob])
            else:
                break
        print(table)

    def generate_bigram_prob_intervals(self, wn_minus_one):
        """
        Generate the probability interval for each bigram where the first value
        in the tuple is equal to wn_minus_one.

        :param wn_minus_one: the base for the bigrams of interest
        """

        # Find all bigrams with the same value for wn_minus_one
        probs_for_bigrams_with_equal_wn_minus_one = [(bigram, self.get_bigram_probability(bigram)) for bigram in self.BIGRAM_PROBABILITIES if bigram[0] == wn_minus_one]

        if len(probs_for_bigrams_with_equal_wn_minus_one) == 0:
            return
        random.shuffle(probs_for_bigrams_with_equal_wn_minus_one)
        prob_intervals = collections.OrderedDict()
        prev_prob = 0

        for bigram_and_prob in probs_for_bigrams_with_equal_wn_minus_one:
            bigram = bigram_and_prob[0]
            current_prob = bigram_and_prob[1]
            prob_intervals[bigram] = (prev_prob, (prev_prob + current_prob))
            prev_prob += current_prob

        self.set_bigram_prob_intervals(wn_minus_one, prob_intervals)

    def sort_bigram_by_count(self):
        """Sort self.BIGRAMS by count"""
        sorted_bigrams_by_count = sorted(self.BIGRAMS.items(),
                                          key=lambda t: t[1],
                                          reverse=True)
        ordered_dict_of_bigram_counts = collections.OrderedDict()

        # Store the sorted bigrams inside of the OrderedDict: ordered_dict_of_bigram_counts
        for bigram_and_count in sorted_bigrams_by_count:
            bigram = bigram_and_count[0]
            observed_sequence = bigram_and_count[1]
            ordered_dict_of_bigram_counts[bigram] = observed_sequence

        self.BIGRAMS = ordered_dict_of_bigram_counts
        self.bigrams_are_sorted = True

    def generate_random_sentence(self):
        """Generate a random sentence using bigram probabilities"""

        end_of_sentence_reached = False
        sentence = ''

        # To start a sentence, <s> is set as the initial previous word.
        prev_word = '<s>'
        # For demonstration purposes, the max is used to prevent long/infinite sentence generation.
        max_words_in_sentence = 100
        count = 0
        sentence_tokens = []
        # Continuously generate random words based off of the previous word.
        # When the </s> token is generated, or the maximum amount of words per sentence (wps)
        # is reached, stop generating words and display the sentence.
        while not end_of_sentence_reached:
            rand_bigram = self.get_random_bigram(prev_word)

            wn = rand_bigram[1]
            if wn == '</s>':  # End of sentence
                end_of_sentence_reached = True
            else:
                if count < max_words_in_sentence:
                    prev_word = wn
                    count += 1
                else:  # Max wps has been reached
                    end_of_sentence_reached = True
                sentence = sentence + str(wn) + ' '
            sentence_tokens.append(rand_bigram)
        print()
        print('GENERATED SENTENCE:')
        print(sentence)
        print()
        self.compute_sentence_probability(sentence_tokens)

    def compute_sentence_probability(self, sentence_tokens):

        sentence_log_probability = 0.0
        for bigram in sentence_tokens:
            bigram_prob = self.get_bigram_probability(bigram)
            bigram_log_probability = math.log(bigram_prob)

            # (P1*P2*P3*P4) == exp(log(P1)*log(P2)*log(P3)*log(P4))
            sentence_log_probability += bigram_log_probability

        print('The probability of generating the sentence above in log scale is: {0:.20f}'.format(sentence_log_probability))

    def generate_bigram(self, token, token_plus_one, start_of_sentence):
        """
        Filter bigram so that punctuation is handeled.
        For example:

            >>> filtered_bigram = generate_bigram(hat, .)
            >>> filtered_bigram
            (hat, </s>)

        :param token, token_plus_one: wn(current word), wn-1
        :rtype: If there is not a second entry to the bigram, return nothing
        """
        if (self.is_punctuation(token)):
            token = '<s>'
        elif self.is_punctuation(token_plus_one):
            token_plus_one = '</s>'

        if token_plus_one is None:
            return

        self.set_bigram((token, token_plus_one))
        if start_of_sentence:
            self.set_bigram(('<s>', token_plus_one))

    def is_punctuation(self, token):
        """
        Return True if the token is a punctuation.

        :param token: wn(current word)
        :rtype: boolean
        """
        return token in['.', '!', '?']

    def print_bigrams(self, k):

        top = k
        if self.bigrams_are_sorted is False:
            self.sort_bigram_by_count()

        print()
        print('Bigram Count: V = {}'.format(len(self.BIGRAMS)))
        table = PrettyTable(['Wn-1', 'Wn', 'C(Wn|Wn-1)'])

        for counter, bigram in enumerate(self.BIGRAMS):
            if counter < top:
                wn = bigram[1]
                wn_minus_one = bigram[0]
                observed_sequence = self.get_bigram_count(bigram)
                table.add_row([wn_minus_one, wn, observed_sequence])
            else:
                break
        print(table)

