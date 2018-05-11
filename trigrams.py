from prettytable import PrettyTable
import collections
import random
import math

class Trigrams():

    # initialize
    def __init__(self):
        self.TRIGRAMS = {}
        self.TRIGRAM_PROBABILITIES = {}
        # The frequency counts for each trigram prefix
        self.FREQUENCY_COUNT = {}
        # Intervals for each trigram probability. Formatted as {(wn_minus_one, Wn) : (low, high)}
        self.TRIGRAM_PROB_INTERVALS = {}

        self.trigrams_are_sorted = False

    def set_trigram(self, trigram):
        """
        Add trigram to TRIGRAMS dict if it does not already exists;
        otherwise, increment count.

        :param trigram: trigram to insert or increment
        """
        if (trigram not in self.TRIGRAMS.keys()):
            self.TRIGRAMS[trigram] = 1
        else:
            self.TRIGRAMS[trigram] += 1

    def set_trigram_probabilities(self):
        """Compute and record the probabilities for every trigram."""
        if self.trigrams_are_sorted is False:
            self.sort_trigram_by_count()

        trigram_probabilities = collections.OrderedDict()
        for x in self.TRIGRAMS.items():
            trigram = x[0]
            wn_minus_one = trigram[1]
            wn_minus_two = trigram[0]
            prev_trigram = (wn_minus_two, wn_minus_one)
            observed_frequency = x[1]
            prob = float(observed_frequency / self.FREQUENCY_COUNT[prev_trigram])
            trigram_probabilities[trigram] = prob

        self.TRIGRAM_PROBABILITIES = trigram_probabilities

    def set_trigram_frequency_counts(self):
        """Compute and return the counts for each trigram prefix"""
        trigram_freq_counts = {}

        for x in self.TRIGRAMS.items():
            trigram = x[0]
            wn_minus_one = trigram[1]
            wn_minus_two = trigram[0]
            prev_trigram = (wn_minus_two, wn_minus_one)
            observed_frequency = x[1]
            if (prev_trigram not in trigram_freq_counts):
                trigram_freq_counts[prev_trigram] = observed_frequency
            else:
                trigram_freq_counts[prev_trigram] += observed_frequency

        self.FREQUENCY_COUNT = trigram_freq_counts

    def set_trigram_prob_intervals(self, prev_bigram, prob_intervals):
        """
        Set probability interval for trigram if it does not exist.

        :param prev_bigram: the base for the trigrams of interest -- will act as the key
        :param prob_intervals: dictionary containing the (trigram, interval) values for
            trigrams with the same wn_minus_one -- will act as value
        """
        if prev_bigram not in self.TRIGRAM_PROB_INTERVALS.keys():
            self.TRIGRAM_PROB_INTERVALS[prev_bigram] = prob_intervals

    def get_trigram_count(self, trigram):
        """
        Return the count for a specific trigram.

        :param trigram: trigram to search for
        :rtype: int
        """
        try:
            return self.TRIGRAMS[trigram]
        except KeyError:
            print('The trigram you have entered , {} , could not be found\n'
                   .format(trigram))
            return -1

    def get_trigram_probability(self, trigram):
        """
        Return the probability for a trigram.

        :param trigram: trigram to search for
        :param trigram_probabilities:
            dictionary containing trigram probabilities

        :rtype: float
        """

        try:
            return self.TRIGRAM_PROBABILITIES[trigram]
        except KeyError:
            print('The trigram you have entered PROB, {} , could not be found\n'
                   .format(trigram))
            return -1

    def get_random_trigram(self, prev_bigram):
        """
        Return a random trigram

        :param prev_word: the base for the trigrams of interest

        :rtype: tuple
        """

        # Prevent duplicate sorting
        if prev_bigram not in self.TRIGRAM_PROB_INTERVALS.keys():
            self.generate_trigram_prob_intervals(prev_bigram)

        # Dictionary containing the intervals for trigrams with the same value for wn_minus_one
        trigrams_with_prev_bigram_intervals = self.get_trigram_prob_intervals(prev_bigram)

        if trigrams_with_prev_bigram_intervals == -1:
            print('There are no trigrams with the prev_bigram you have entered, {}.\n'
                   .format(prev_bigram))
            return -1

        rand_num = random.uniform(0.0, 1.0)
        rand_trigram = ''

        # Find the trigram where rand_num falls between its interval
        for trigram in trigrams_with_prev_bigram_intervals:
            interval = trigrams_with_prev_bigram_intervals[trigram]
            low = interval[0]  # lower boundary
            high = interval[1]  # upper boundary

            if low < rand_num <= high:  # random word is found
                rand_trigram = trigram
                break

        return rand_trigram

    def get_trigram_prob_intervals(self, prev_bigram):
        """
        Get the probability interval for trigrams where (Wn-2, Wn-1) equals prev_bigram

        :param prev_bigram: the base for the trigrams of interest

        :rtype: dict
        """
        try:
            return self.TRIGRAM_PROB_INTERVALS[prev_bigram]
        except KeyError:
           # print('There are no trigrams with the prev_bigram you have entered, {}.\n'
           #        .format(prev_bigram))
            return -1

    def get_trigrams_for_given_pre_bigram(self, prev_bigram, k):
        if prev_bigram not in self.TRIGRAM_PROB_INTERVALS.keys():
            self.generate_trigram_prob_intervals(prev_bigram)

        if self.get_trigram_prob_intervals(prev_bigram) == -1:
            return

        table = PrettyTable(['(Wn-2, Wn-1, Wn)', 'C(Wn-2, Wn-1, Wn)', 'P(Wn-2, Wn-1, Wn)'])

        dict_of_trigram_and_counts = {}

        for trigram in self.TRIGRAM_PROB_INTERVALS[prev_bigram]:
            dict_of_trigram_and_counts[trigram] = self.get_trigram_count(trigram)

        trigrams_sorted_by_count = sorted(dict_of_trigram_and_counts, key=lambda x: dict_of_trigram_and_counts[x], reverse=True)

        try:
            for counter, trigram in enumerate(trigrams_sorted_by_count):
                if counter < k:
                    trigram_count = self.get_trigram_count(trigram)
                    prob = self.get_trigram_probability(trigram)
                    prob = '{:.6f}'.format(prob)
                    table.add_row([trigram, trigram_count, prob])
                else:
                    break
            print(table)
        except KeyError:
            print()
            print('There are no trigrams matching the (Wn-2, Wn-1) you have entered: {}'
                    .format(prev_bigram))

    def generate_trigram_prob_intervals(self, prev_bigram):
        """
        Generate the probability interval for each trigram where (Wn-2, Wn-1)
        in the tuple is equal to prev_bigram.

        :param prev_bigram: the base for the trigrams of interest
        """

        # Find all trigrams where (Wn-2, Wn-1) equals prev_bigram
        probs_for_trigrams_with_equal_prev_bigram = [(x,self.get_trigram_probability(x)) for x in self.TRIGRAM_PROBABILITIES if (x[0], x[1]) == prev_bigram]

        if len(probs_for_trigrams_with_equal_prev_bigram) == 0:
            return

        random.shuffle(probs_for_trigrams_with_equal_prev_bigram)
        prob_intervals = collections.OrderedDict()
        prev_prob = 0

        for trigram_and_prob in probs_for_trigrams_with_equal_prev_bigram:
            trigram = trigram_and_prob[0]
            current_prob = trigram_and_prob[1]
            prob_intervals[trigram] = (prev_prob, (prev_prob + current_prob))
            prev_prob += current_prob

        self.set_trigram_prob_intervals(prev_bigram, prob_intervals)

    def sort_trigram_by_count(self):
        """Sort self.TRIGRAMS by count"""
        sorted_trigrams_by_count = sorted(self.TRIGRAMS.items(),
                                          key=lambda t: t[1],
                                          reverse=True)
        ordered_dict_of_trigram_counts = collections.OrderedDict()

        # Store the sorted trigrams inside of the OrderedDict: ordered_dict_of_trigram_counts
        for trigram_and_count in sorted_trigrams_by_count:
            trigram = trigram_and_count[0]
            observed_sequence = trigram_and_count[1]
            ordered_dict_of_trigram_counts[trigram] = observed_sequence

        self.TRIGRAMS = ordered_dict_of_trigram_counts
        self.trigrams_are_sorted = True

    def generate_random_sentence(self):
        """Generate a random sentence using trigram probabilities"""

        end_of_sentence_reached = False
        sentence = ''

        # To start a sentence, <s> is set as the initial previous word.
        prev_bigram = ('<s>', '<s>')
        # For demonstration purposes, the max is used to prevent long/infinite sentence generation.
        max_words_in_sentence = 100
        count = 0
        sentence_tokens = []
        # Continuously generate random words based off of the previous bigram.
        # When the </s> token is generated, or the maximum amount of words per sentence (wps)
        # is reached, stop generating words and display the sentence.
        while not end_of_sentence_reached:
            rand_trigram = self.get_random_trigram(prev_bigram)

            if rand_trigram == -1:
                break
            wn = rand_trigram[2]
            wn_minus_one = rand_trigram[1]
            if wn == '</s>':  # End of sentence
                end_of_sentence_reached = True
            else:
                if count < max_words_in_sentence:
                    sentence = sentence + wn + ' '
                    prev_bigram = (wn_minus_one, wn)
                    count += 1
                else:  # Max wps has been reached
                    print('Max reached')
                    end_of_sentence_reached = True
            sentence_tokens.append(rand_trigram)

        print()
        print('GENERATED SENTENCE:')
        print(sentence)
        print()
        self.compute_sentence_probability(sentence_tokens)

    def generate_trigram(self, token, token_plus_one, token_plus_two, start_of_sentence):
        """
        Filter trigram so that punctuation is handeled.
        For example:
            >>> filtered_trigram = generate_trigram(the, hat, .)
            >>> filtered_trigram
            (the, hat, </s>)

        :rtype: If there is not a second or third entry to the trigram, return nothing
        """
        if token_plus_one is None or token_plus_two is None:
            return

        if self.is_punctuation(token_plus_one):
            return

        if self.is_punctuation(token_plus_two):
            token_plus_two = '</s>'
        elif self.is_punctuation(token):
            token = '<s>'

        self.set_trigram((token, token_plus_one, token_plus_two))
        if start_of_sentence:
            self.set_trigram(('<s>', '<s>', token_plus_one))

    def compute_sentence_probability(self, sentence_tokens):

        sentence_log_probability = 0.0
        for trigram in sentence_tokens:
            trigram_log_probability = 0
            trigram_prob = self.get_trigram_probability(trigram)
            if trigram_prob != -1:
                trigram_log_probability = math.log(trigram_prob)

            # (P1*P2*P3*P4) == exp(log(P1)+log(P2)+log(P3)+log(P4))
            sentence_log_probability += trigram_log_probability

        print('The probability of generating the sentence above in log scale is: {0:.20f}'.format(sentence_log_probability))

    def is_punctuation(self, token):
        """
        Return True if the token is a punctuation.

        :param token: wn(current word)
        :rtype: boolean
        """
        return token in['.', '!', '?']

    def print_trigrams(self, k):

        top = k
        if self.trigrams_are_sorted is False:
            self.sort_trigram_by_count()

        print()
        print('Trigram Count: V = {}'.format(len(self.TRIGRAMS)))
        table = PrettyTable(['Wn-2', 'Wn-1', 'Wn', 'C(Wn|Wn-2Wn-1)'])

        for counter, trigram in enumerate(self.TRIGRAMS):
            if counter < top:
                wn = trigram[2]
                wn_minus_one = trigram[1]
                wn_minus_two = trigram[0]
                observed_sequence = self.get_trigram_count(trigram)
                table.add_row([wn_minus_two, wn_minus_one, wn, observed_sequence])
            else:
                break

        print(table)

