"""
API for ex2, implementing the skip-gram model (with negative sampling).

"""
import pickle
import os
import time
import re
import random
import sys
import math
import collections
import nltk
import numpy as np
import pandas as pd

from collections import defaultdict
from nltk.corpus import stopwords
from numpy.linalg import norm

nltk.download('stopwords')
random.seed(20864556)


################## Static Functions #################################

def who_am_i():  # this is not a class method
    """
    Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
    """

    return {'name': 'Tal Hayun', 'id': '208645564', 'email': 'hayunta@post.bgu.ac.il'}


def normalize_text(fn):
    """
    Loading a text file and normalizing it, returning a list of sentences.

    Args:
        fn: full path to the text file to process.

    Returns:
        A list where each sublist contains the tokenized words of a sentence from the processed text file.
    """

    with open(fn, 'r') as file:
        text = file.read()

    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]'“'^_`{|}~"

    # Replace newlines with a 'EOS' marker that will later define sentence boundaries
    text = text.replace('\n', ' eos ')

    # Replace all non-alphanumeric characters with spaces, except for periods.
    text = re.sub(r'\s+', ' ', text).strip().lower()  # Remove line breaks, extra spaces, and make it lowercase

    sentences = text.split('eos ')

    # Trim any leading or trailing whitespace that might have been added
    tokenized_sentences = [
        [word.strip(punctuation) for word in filter(None, sentence.strip().split())]
        for sentence in sentences if sentence.strip()
    ]

    return tokenized_sentences


def sigmoid(x): return 1.0 / (1 + np.exp(-x))


def save_model(self, fn):
    """
    Saves the entire object to a file using Python's pickle module.

    Args:
    fn (str): The filename or path where the object should be saved. If the path does not exist,
              a `FileNotFoundError` may be raised. The file will be written in binary mode.
    """
    with open(fn, 'wb') as file:
        pickle.dump(self, file)


def load_model(fn):
    """
    Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.

    Returns:
        The SGNS model instance that loaded from pickle file.
    """

    with open(fn, 'rb') as file:
        sg_model = pickle.load(file)
    return sg_model

################## Loading Corpora Functions #################################

def load_dataset(name):
    """
    Loads a dataset from a text file and returns its content as normalized sentences.

    Args:
        name (str): The name of the dataset to load. Accepted values are 'harry_potter', 'big', and 'drSuess'

    Returns:
        A list of sentences extracted and normalized from the specified text file.
        This list represents the textual content of the dataset, divided into sentences that have been processed for
        common text normalization task.
    """
    dir_path = os.getcwd()
    if name == 'harry_potter':
        corpus_path = os.path.join(dir_path, 'harryPotter1.txt')
    elif name == 'big':
        corpus_path = os.path.join(dir_path, 'big.txt')
    elif 'drSuess':
        corpus_path = os.path.join(dir_path, 'drSeuss.txt')
    else:
        raise ValueError(f"Error: The dataset at {name} was not found.")

    sentences = normalize_text(corpus_path)
    return sentences


#####################################################################
#                   Skip Gram Class                                 #
#####################################################################

def word_counter(sentences, threshold):
    """
    Counts occurrences of each word across all sentences and filters by a minimum frequency threshold.

    Args:
        sentences (list of list of str): A list where each element is a list of tokens representing a sentence.
        threshold (int): The minimum number of occurrences for a word to be included in the returned dictionary.

     Returns:
        dict: A dictionary where the keys are words and the values are the number of occurrences of each word across
              all input sentences, filtered to include only those words whose occurrence count meets or exceeds
              the specified `threshold`.
    """

    word_counts = defaultdict(int)
    combined_text = ' '.join([' '.join(tokens) for tokens in sentences])
    words = re.findall(r"\b\w+\b", combined_text)

    for word in words:
        word_counts[word] += 1

    words_counts = dict(word_counts)
    filtered_word_counts = {word: count for word, count in word_counts.items() if count > threshold}
    return filtered_word_counts

class SkipGram:
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):
        self.sentences = sentences # List of corpus sentences
        self.d = d  # Embedding dimension
        self.neg_samples = neg_samples  # Num of negative samples for one positive sample
        self.context = context  # The size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold  # Ignore low frequency words (appearing under the threshold)

        # A dictionary that contains word:count - without low frequency
        self.Words = word_counter(sentences, word_count_threshold)
        self.voc = [] # The vocabulary of the given corpus
        self.T, self.C, self.E = None, None, None  # Target Matrix, Context matrix, Embedding matrix

    def get_positive_words(self, word_target):
        """
        Identifies positive words around a target word within a specified window size in each sentence.

        Args:
            word_target (str): The target word around which the context window of words is considered.

        Returns:
             A list of unique positive words around the target word across all sentences.
        """

        stop_words = stopwords.words('english')
        window_size = self.context
        sentences = self.sentences

        positive_words = []
        for sentence in sentences:
            if word_target in sentence:
                target_index = sentence.index(word_target)  # get the index of target in sentence
                start = max(0, target_index - window_size // 2)
                end = min(len(sentence), target_index + window_size // 2 + 1)

                words_before = sentence[start:target_index]
                words_after = sentence[target_index + 1:end]

                positive_words += words_before + words_after

        positive_words_without_stop = list(set(positive_words) - set(stop_words))
        positive_words_without_stop = [word for word in positive_words_without_stop if word in self.Words]
        return positive_words_without_stop

    def get_positive_negative_words(self, word_target):
        """
        Fetches positive and negative word samples related to a target word.

        Args:
            word_target (str): The target word for which positive and negative samples are to be fetched.

        Returns:
                A tuple containing two lists:
                - First list contains positive words associated with the target word.
                - Second list contains negative word samples.
        """

        n_negative = self.neg_samples
        positive_words = self.get_positive_words(word_target)

        stop_words = stopwords.words('english')
        number_of_negative = n_negative * len(positive_words)  # number of samples from negative words
        set_negative_words = set(self.Words) - set(positive_words) - set(stop_words)

        try:
            negative_words = random.sample(set_negative_words,
                                           number_of_negative)  # Sample negative words from negative list
            return positive_words, negative_words
        except ValueError:
            return [], []

    def compute_similarity(self, w1, w2):
        """
        Returns the cosine similarity (in [0,1]) between the specified words.

        Args:
            w1: a word
            w2: a word

        Returns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
        """

        sim = 0.0  # default
        vec1_idx, vec2_idx = self.voc.index(w1), self.voc.index(w2)
        vec_1, vec_2 = self.E[vec1_idx], self.E[vec2_idx]

        if len(vec_1) != len(vec_2):
            raise ValueError("Vectors embedding are not of the same length")

        # Compute the cosine similarity
        dot_product = np.dot(vec_1, vec_2)
        norm_vec1, norm_vec2 = np.linalg.norm(vec_1), np.linalg.norm(vec_2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            raise ValueError(
                "One of the vectors is zero, which causes division by zero in cosine similarity calculation.")

        sim = dot_product / (norm_vec1 * norm_vec2)
        return sim

    def get_closest_words(self, w, n=5):
        """
        Returns a list containing the n words that are the closest to the specified word.

        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.

        Returns:
            A list of the top `n` words that are most similar to the target word.
        """

        similarities_lst = []
        words = list(self.voc)

        for w2 in words:
            if w != w2:
                sim = self.compute_similarity(w, w2)
                similarities_lst.append(sim)
        top_indices = np.argsort(similarities_lst)[-n:]  # The indexes of n closest words

        n_closest_words = [words[index] for index in top_indices]
        return n_closest_words

    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None):
        """
        Returns a trained embedding models and saves it in the specified path.

        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.

        Returns:
            A tuple containing two matrices, T and C, which represent the embedding matrices for the target and context
            words, after the completion of the training process.
        """

        words = self.Words
        stop_words = stopwords.words('english')

        voc = list(set(words) - set(stop_words))
        vocab_size = len(voc)
        self.voc = voc

        C, T = np.random.rand(vocab_size, self.d), np.random.rand(self.d, vocab_size)
        self.C, self.T = C, T

        for epoch in range(epochs):
            for token in voc: # Iterate each token in the vocabilary (in this loop the token is the target)
                pos_words, neg_words = self.get_positive_negative_words(token)

                index_target = voc.index(token)  # Index of target in BOW
                w_target = self.T.T[index_target]  # Weight of target in T matrix

                p_gradients, n_gradients = [], []  # List of positive gradients, List of negative gradients

                for positive_word in pos_words:
                    index_positive = voc.index(positive_word)  # Index of Positive word in BOW
                    w_positive = C[index_positive]  # Weight of positive word in C matrix

                    gradient_pos = (sigmoid(np.dot(w_positive, w_target)) - 1) * w_positive
                    p_gradients.append(gradient_pos)
                    w_positive = w_positive - step_size * gradient_pos  # Update positive Weight
                    self.C[index_positive] = w_positive

                for negative_word in neg_words:
                    index_negative = voc.index(negative_word)  # Index of Negative word in BOW
                    w_negative = C[index_negative]  # Weight of negative word in C matrix

                    gradient_neg = (sigmoid(np.dot(w_negative, w_target))) * w_negative
                    n_gradients.append(gradient_neg)
                    w_negative = w_negative - step_size * gradient_neg
                    self.C[index_negative] = w_negative  # Update negative sample weight into C matrix

                sum_gradients = step_size * (np.sum(p_gradients) + np.sum(n_gradients))
                self.T.T[index_target] = w_target - sum_gradients  # Update target weight into T matrix
            # print(f'Finish epoch no.{epoch} ({epoch} out of {epochs})')

        # Save the embedding matrix into path
        embedding_path = os.path.join(os.getcwd(), 'embedding_mtx.pkl') # Create path to save the embedding matrix
        self.E = self.combine_vectors(T, C, 2, embedding_path) # Update the Embedding matrix

        # Save model into path
        save_model(self, model_path)

        return T, C

    def combine_vectors(self, T, C, combo=0, model_path=None):
        """
        Returns a single embedding matrix and saves it to the specified path

        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how wo combine the T and C embeddings (int)
                   0: use only the T embeddings (default)
                   1: use only the C embeddings
                   2: return a pointwise average of C and T
                   3: return the sum of C and T
                   4: concat C and T vectors (effectively doubling the dimension of the embedding space)
            model_path: full path (including file name) to save the model pickle at.

        Returns:
            A single embedding matrix resulting from the combination of the target (T) and context (C) embeddings
            based on the specified `combo` method.
        """

        if combo == 0: # Only the T embeddings
            embedding_mtx =  self.T.T
        elif combo == 1: # Only the C embeddings
            embedding_mtx = self.C
        elif combo == 2: # Average of C and T
            embedding_mtx = (self.T.T + self.C)/2
        elif combo == 3: # Sum of C and T
            embedding_mtx = (self.T.T + self.C)
        elif combo == 4: # Concat C and T vectors
            embedding_mtx = np.concatenate((self.C, self.T.T), axis=1)
        else:
            raise ValueError("Combo range should between 0 to 4 (included)")

        # Save embedding matrix into pickle file and returns the matrix
        with open(model_path, 'wb') as file:
            pickle.dump(embedding_mtx, file)
        return embedding_mtx

    def find_analogy(self, w1, w2, w3):
        """
        Returns a word (string) that matches the analogy test given the three specified words.
        Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)

        Returns:
            The word that best completes the analogy: "w1 is to w2 as w3 is to ____."
        """

        if w1 not in self.voc or w2 not in self.voc or w3 not in self.voc:
            return "One or more words not in BOW."

        w1_idx, w2_idx, w3_idx = self.voc.index(w1), self.voc.index(w2), self.voc.index(w3)
        w1_embedding, w2_embedding, w3_embedding = self.E[w1_idx], self.E[w2_idx], self.E[w3_idx]

        w4_embedding = w1_embedding - w2_embedding + w3_embedding # compute new vector

        best_sim, w = -1, None # Define best similarity and best word (w)
        for word in self.voc:
            if word not in [w1, w2, w3]:
                word_idx = self.voc.index(word)
                word_embedding = self.E[word_idx]

                sim = np.dot(word_embedding, w4_embedding) / (norm(word_embedding) * norm(w4_embedding))  # Cosine similarity
                best_sim, w = (sim, word) if sim > best_sim else (best_sim, w)
        return w

    def test_analogy(self, w1, w2, w3, w4, n=1):
        """
        Returns True if sim(w1-w2+w3, w4)@n; Otherwise return False.
            That is, returning True if w4 is one of the n closest words to the vector w1-w2+w3.
            Interpretation: 'w1 to w2 is like w4 to w3'

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
             w4: forth word in the analogy (string)
             n: the distance (work rank) to be accepted as similarity
        """

        w1_idx, w2_idx, w3_idx = self.voc.index(w1), self.voc.index(w2), self.voc.index(w3)
        agg_vec = self.E[w1_idx] - self.E[w2_idx] + self.E[w3_idx]

        sim_words = []
        for word in self.voc :
            if word not in [w1, w2, w3]:
                word_idx = self.voc.index(word)
                word_embedding = self.E[word_idx]

                cosine = np.dot(agg_vec, word_embedding) / (norm(agg_vec) * norm(word_embedding)) # Cosine similarity
                sim_words.append(cosine)

        top_indices = sorted(range(len(sim_words)), key=lambda x: sim_words[x], reverse=True)[:n] # Select top n indexes words
        top_closest_words = [self.voc[index] for index in top_indices] # Get list of the words
        return True if w4 in top_closest_words else False

################## Tesets function #################################

def tests(model):
    # Harry Potter Corpus -
    # Skip Gram Hyperparameters: d=200, neg_samples=4, context=4, word_count_threshold=5)
    # Learn Embedding Hyperparameters: step_size=0.001, epochs=200, early_stopping=3
    assert model.test_analogy('hogwarts', 'magic', 'muggle', 'classroom', n=2)
    assert model.test_analogy('goblin', 'person', 'stone', 'griphook', n=3)
    assert model.test_analogy('family', 'wizard', 'muggles', 'dudley', n=7)
    assert model.test_analogy('team', 'students', 'wizards', 'hogwarts', n=14)

    assert model1.find_analogy("hogwarts", "wizards", "students") == 'twenty'
    assert model1.find_analogy('harry', 'famous', 'hermione') == 'sharp'


if __name__ == '__main__':
    # Load the corpora
    corpus_sentences = load_dataset("harry_potter")
    # Create SG model
    sg1 = SkipGram(sentences=corpus_sentences, d=200, neg_samples=4, context=4, word_count_threshold=5)

    # The path that the instance will be saved
    model_path = os.path.join(os.getcwd(), 'model.pkl')
    # Learn Embeddings
    sg1.learn_embeddings(step_size=0.001, epochs=200, early_stopping=3, model_path=model_path)
    # Load the model from pickle file
    model1 = load_model(model_path)

    # Tests the results by 'test_analogy' and 'find_analogy' methods
    tests(model1)