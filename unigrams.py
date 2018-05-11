from prettytable import PrettyTable
import random
import collections
import math

class Unigrams():

    # initialize
    def __init__(self):
        self.UNIGRAMS = {}
        self.UNIGRAM_PROBABILITIES = {}
        # Equivalent to length of tokens inside of text
        self.UNIGRAM_COUNT_TOTAL = 0
        # Intervals for each unigram probability. Formatted as {Wn : (low, high)}
        self.UNIGRAM_PROB_INTERVALS = {}

        # Prevent unnecessary sorting
        self.unigrams_are_sorted = False
        self.unigram_intervals_are_recorded = False

    def set_unigram(self, unigram):
        """
        Add unigram to UNIGRAMS dict if it does not already exists;
        otherwise, increment count.

        :param unigram: unigram to insert or increment
        """
        if (unigram not in self.UNIGRAMS.keys()):
            self.UNIGRAMS[unigram] = 1
        else:
            self.UNIGRAMS[unigram] += 1

    def set_unigram_probabilities(self):
        """Compute the probabilities for each unigram."""
        if self.unigrams_are_sorted is False:
            self.sort_unigram_by_count()
        unigram_probabilities = collections.OrderedDict()

        for unigram in self.UNIGRAMS.items():
            wn = unigram[0]
            wn_observed_frequency = unigram[1]
            probability = float(wn_observed_frequency / self.UNIGRAM_COUNT_TOTAL)
            unigram_probabilities[wn] = probability

        self.UNIGRAM_PROBABILITIES = unigram_probabilities

    def set_unigram_count_total(self):
        """Record the count for all unigrams"""
        count = 0

        for unigram in self.UNIGRAMS.items():
            observed_frequency = unigram[1]
            count += observed_frequency
        self.UNIGRAM_COUNT_TOTAL = count

    def set_unigram_prob_intervals(self):
        """
        Record the interval for each unigram. This interval will be
        used to generate random sentences
        """
        probability_intervals = {}
        prev_prob = 0

        for wn in self.UNIGRAM_PROBABILITIES:
            current_prob = self.get_unigram_probability(wn)
            probability_intervals[wn] = (prev_prob, (prev_prob + current_prob))
            prev_prob += current_prob

        self.UNIGRAM_PROB_INTERVALS = probability_intervals

    def get_unigram_count(self, unigram):
        """
        Return the count for a specific unigram.

        :param unigram: unigram to search for
        :rtype: int
        """
        try:
            return self.UNIGRAMS[unigram]
        except KeyError:
            print('The unigram you have entered, {} , could not be found\n'
                   .format(unigram))
            return -1

    def get_unigram_probability(self, unigram):
        """
        Return the probability for a unigram.

        :param unigram: unigram to search for

        :rtype: float
        """
        try:
            return self.UNIGRAM_PROBABILITIES[unigram]
        except KeyError:
            print('The unigram you have entered, {} , could not be found\n'
                   .format(unigram))
            return -1

    def get_random_unigram(self):
        """
        Return a random unigram

        :rtype: str
        """
        rand_num = random.uniform(0.0, 1.0)
        rand_unigram = self.get_wn_of_enclosed_interval(rand_num)
        return rand_unigram

    def get_unigram_count_total(self):
        """
        Return the count for all unigrams

        :rtype: int
        """
        return self.UNIGRAM_COUNT_TOTAL

    def get_wn_of_enclosed_interval(self, rand_num):
        """
        Get the interval that encloses rand_num.

        :param rand_num: a random float between 0.0 and 1.0

        :rtype: str
        """
        wn = ''
        for key, value in self.UNIGRAM_PROB_INTERVALS.items():
            interval = value
            low = interval[0]
            high = interval[1]
            if low < rand_num and rand_num < high:
                wn = key
        return wn

    def generate_random_sentence(self):
        """Generate a random sentence using unigram probabilities"""
        if self.unigram_intervals_are_recorded is False:
            self.set_unigram_prob_intervals()

        sentence = ''
        end_of_sentence_reached = False

        # For demonstration purposes, the max is used to prevent long/infinite sentence generation.
        max_words_in_sentence = 100
        count = 0
        sentence_tokens = []
        # Continuously generate random words based off of the previous word.
        # When the </s> token is generated, or the maximum amount of words per sentence (wps)
        # is reached, stop generating words and display the sentence.
        while not end_of_sentence_reached:
            rand_unigram = self.get_random_unigram()

            if rand_unigram == '</s>':  # End of sentence
                end_of_sentence_reached = True
            else:
                if count < max_words_in_sentence:
                    sentence = sentence + rand_unigram + ' '
                    count += 1
                else:  # Max wps has been reached
                    end_of_sentence_reached = True
            sentence_tokens.append(rand_unigram)
        print()
        print('GENERATED SENTENCE:')
        print(sentence)
        print()

        self.compute_sentence_probability(sentence_tokens)

    def compute_sentence_probability(self, sentence_tokens):
        """
        Compute the probability of a sentence occurring.

        :param sentence_tokens: A list of the tokens of a sentence"
        """
        sentence_log_probability = 0.0

        for unigram in sentence_tokens:
            unigram_prob = self.get_unigram_probability(unigram)
            unigram_log_probability = math.log(unigram_prob)

            # (P1*P2*P3*P4) == exp(log(P1)*log(P2)*log(P3)*log(P4))
            sentence_log_probability += unigram_log_probability

        print('The probability of generating the sentence above in log scale is: {0:.20f}'.format(sentence_log_probability))

    def sort_unigram_by_count(self):
        """Sort self.UNIGRAMS by count"""
        sorted_unigrams_by_count = sorted(self.UNIGRAMS.items(),
                                          key=lambda t: t[1],
                                          reverse=True)

        ordered_dict_of_unigram_counts = collections.OrderedDict()

        # Store the sorted unigrams inside of the OrderedDict: ordered_dict_of_unigram_counts
        for unigram_and_count in sorted_unigrams_by_count:
            unigram = unigram_and_count[0]
            observed_sequence = unigram_and_count[1]
            ordered_dict_of_unigram_counts[unigram] = observed_sequence

        self.UNIGRAMS = ordered_dict_of_unigram_counts
        self.unigrams_are_sorted = True

    def generate_unigram(self, token):
        """
        Filter unigram so that punctuation is handeled.
        For example:

            >>> filtered_unigram = filter_unigram((.))
            >>> filtered_unigram
            (</s>)

        :param token: wn

        """
        if (self.is_punctuation(token)):
            token = '</s>'

        self.set_unigram(token)

    def is_punctuation(self, token):
        """
        Return True if the token is a punctuation.

        :param token: wn(current word)
        :rtype: boolean
        """
        return token in['.', '!', '?']

    def print_unigrams(self, k):

        top = k

        # Prevent unnecessary sorting.
        if self.unigrams_are_sorted is False:
            self.sort_unigram_by_count()

        print()
        print('Unigram Count: V = {}'.format(len(self.UNIGRAMS)))
        table = PrettyTable(['Wn', 'C(Wn)', 'P(Wn)'])
        for counter, unigram in enumerate(self.UNIGRAMS):
            if counter < top:
                observed_sequence = self.get_unigram_count(unigram)
                prob = self.get_unigram_probability(unigram)
                prob = '{0:.6f}'.format(prob)
                table.add_row([unigram, observed_sequence, prob])
            else:
                break
        print(table)


