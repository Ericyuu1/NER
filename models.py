# models.py

from optimizers import *
from nerdata import *
from utils import *
from tqdm import tqdm
from collections import Counter
from typing import List

import numpy as np
import time
import os


class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.
    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]

class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def decode(self, sentence_tokens: List[Token])->LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        scorer=ProbabilisticSequenceScorer(self.tag_indexer,self.word_indexer,self.init_log_probs,self.transition_log_probs,self.emission_log_probs)
        pred_tags=[]
        T=len(sentence_tokens)
        N=len(self.tag_indexer)
        viterbi=np.zeros(shape=(N,T))
        backpointer=np.zeros(shape=(N,T))
        #referenced from https://stanford.edu/~jurafsky/slp3/A.pdf
        #Initialization
        for s in range(N):
            viterbi[s, 0] = scorer.score_init(sentence_tokens, s) + scorer.score_emission(sentence_tokens, s, 0)
            backpointer[s,0]=0
        # Recursion
        for t in range(1, T):
            for s in range(N):
                vite = np.zeros(N)
                bpt = np.zeros(N)
                for sft in range(N):
                    vite[sft] = viterbi[sft, t - 1] + scorer.score_transition(sentence_tokens, sft,s) + scorer.score_emission(sentence_tokens,s,t)
                    bpt[sft] = viterbi[sft, t - 1] + scorer.score_transition(sentence_tokens, sft,s)
                viterbi[s, t] = np.max(vite)
                backpointer[s, t] = np.argmax(bpt)
        # Backtrace
        pred_tags.append(self.tag_indexer.get_object(np.argmax(viterbi[:, T - 1])))
        for t in range(1, T):
            pred_tags.append(self.tag_indexer.get_object(backpointer[self.tag_indexer.index_of(pred_tags[-1]), T - t]))
        pred_tags = list(reversed(pred_tags))
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))



def viterbi(sentence_tokens: List[Token], scorer: ProbabilisticSequenceScorer)->LabeledSentence:
    N = len(sentence_tokens)
    T = len(scorer.tag_indexer)
    v = np.zeros((N, T)) 
    y_pred = np.zeros((N, T))
    # Initial states
    for y in range(T):
        # prob of having state y,
        v[0, y] = scorer.score_init(sentence_tokens, y) + scorer.score_emission(sentence_tokens, y, 0)

    for i in range(1, N):
        for y in range(T):
            previous_prob = np.zeros(T)
            for y_prev in range(T):
                previous_prob[y_prev] = scorer.score_transition(sentence_tokens, y_prev, y) + v[i-1, y_prev]
            v[i, y] = scorer.score_emission(sentence_tokens, y, i) + np.max(previous_prob)
            y_pred[i, y] = np.argmax(previous_prob)

    idx = int(np.argmax(v[-1, :]))
    pred_tags = [(scorer.tag_indexer.get_object(idx))]
    for t in range(1, N):
        idx = int(y_pred[N - t, idx])
        pred_tags.append(scorer.tag_indexer.get_object(idx))
    pred_tags.reverse()
    return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))


def train_hmm_model(sentences: List[LabeledSentence]) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)

    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer), len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer), len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print("Tag indexer: %s" % tag_indexer)
    print("Initial state log probabilities: %s" % init_counts)
    print("Transition log probabilities: %s" % transition_counts)
    print("Emission log probs too big to print...")
    print("Emission log probs for India: %s" % emission_counts[:,word_indexer.add_and_get_index("India")])
    print("Emission log probs for Phil: %s" % emission_counts[:,word_indexer.add_and_get_index("Phil")])
    print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.weights = weights

    def decode(self, sentence_tokens: List[Token])->LabeledSentence:
        pred_tags = []
        N = len(sentence_tokens)
        T = len(self.tag_indexer)
        v = np.zeros(shape=(T, N))
        max_pred = np.zeros(shape=(T, N))
        score_matrix = np.zeros(shape=(T, N))
        for y in range(T):
            for i in range(N):
                features = extract_emission_features(sentence_tokens,i,self.tag_indexer.get_object(y),self.feature_indexer,add_to_indexer=False)
                score = sum([self.weights[i] for i in features])
                score_matrix[y, i] = score
        # Initialization
        for y in range(T):
            tag = str(self.tag_indexer.get_object(y))
            if (isI(tag)):
                v[y, 0] = float("-inf")
            else:
                v[y, 0] = score_matrix[y, 0]
            max_pred[y, 0] = 0
        # Recursion
        for i in range(1, N):
            for y in range(T):
                prev_prob = np.zeros(T)
                for y_prev in range(T):
                    prev_tag = str(self.tag_indexer.get_object(y_prev))
                    curr_tag = str(self.tag_indexer.get_object(y))
                    if (isO(prev_tag) and isI(curr_tag)) or (isI(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)) or (isB(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)):
                        prev_prob[y_prev] = float("-inf")
                    else:
                        prev_prob[y_prev] = v[y_prev, i - 1]
                v[y, i] = np.max(prev_prob) + score_matrix[y, i]
                max_pred[y, i] = np.argmax(prev_prob)
        # Backtrace
        pred_tags.append(self.tag_indexer.get_object(np.argmax(v[:, N - 1])))
        for i in range(1, N):
            pred_tags.append(self.tag_indexer.get_object(max_pred[self.tag_indexer.index_of(pred_tags[-1]), N - i]))
        pred_tags = list(reversed(pred_tags))
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))


# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences, run_experiments=False):
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
    print("Training")

    sentence_num = int(len(sentences))
    weights = np.random.rand(len(feature_indexer))
    optimizer = SGDOptimizer(weights, 0.1)
    epoch = 20
    for i in tqdm(range(epoch)):
        loss = 0
        start = time.time()
        train_index = np.arange(sentence_num)
        np.random.shuffle(train_index)
        for sentence_idx in train_index:
            gradients = Counter()

            N = len(sentences[sentence_idx])
            T = len(tag_indexer)

            #feature matrix
            feature_matrix = np.zeros(shape=(T, N))
            for y in range(T):
                for x in range(N):
                    feature_matrix[y, x] = np.sum(np.take(weights, feature_cache[sentence_idx][x][y]))

            forward = np.zeros(shape=(T, N)) 
            backward = np.zeros(shape=(T, N))
            #   Forward-backward algorithm
            # Initialization
            for y in range(T):
                forward[y, 0] = feature_matrix[y, 0]

            # Recursion
            for x in range(1, N):
                for y in range(T):
                    sum = 0
                    for y_prev in range(T):
                        if y_prev == 0:
                            sum = forward[y_prev, x - 1]
                        else:
                            sum = np.logaddexp(sum, forward[y_prev, x - 1])
                    forward[y, x] = feature_matrix[y, x] + sum

            # Initialization 
            for y in range(T):
                backward[y, N - 1] = 0

            # Recursion
            for x in range(1, N):
                for y in range(T):
                    sum = 0
                    for y_prev in range(T):
                        if y_prev == 0:
                            sum = backward[y_prev, N - x] + feature_matrix[y_prev, N - x]
                        else:
                            sum = np.logaddexp(sum, backward[y_prev, N - x] + feature_matrix[y_prev, N - x])
                    backward[y, N - x - 1] = sum
            Z = 0
            for y in range(T):
                if y == 0:
                    Z = forward[y, -1]
                else:
                    Z = np.logaddexp(Z, forward[y, -1])

            # Compute the posterior probability
            pp = np.zeros(shape=(T, N))
            for y in range(T):
                for x in range(N):
                    pp[y, x] = np.exp(forward[y, x] + backward[y, x] - Z)

            #  Compute the gradient of the feature vector for a sentence
            for word_idx in range(len(sentences[sentence_idx])):
                gold_tag = tag_indexer.index_of(sentences[sentence_idx].get_bio_tags()[word_idx])
                features = feature_cache[sentence_idx][word_idx][gold_tag]
                loss += np.sum([weights[i] for i in features])
                for feature in features:
                    gradients[feature] += 1
                for tag_idx in range(T):
                    features = feature_cache[sentence_idx][word_idx][tag_idx]
                    for feature in features:
                        gradients[feature] -= pp[tag_idx, word_idx]

            # Update the weights using the gradient
            loss -= Z
            optimizer.apply_gradient_update(gradients, 10)

        # Calculate the usage of time.
        elapsed_time = time.time() - start
        minutes, seconds = divmod(elapsed_time, 60)
        print('epoch: {} time: {:0>2}:{} loss: {}'.format(i, int(minutes), int(seconds), -loss))
    '''
    list of loss has been deleted.
    plt.plot(x,loss,label="Step=0.1")
    plt.legend(loc='upper right')
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.xticks(np.linspace(0,19,20))
    plt.show()
    '''
    '''
    list of F1 score has been deleted.
    plt.plot(x,loss,label="Step=0.1")
    plt.legend(loc='upper right')
    plt.xlabel("Batch")
    plt.ylabel("F1 Score")
    plt.xticks(np.linspace(0,19,20))
    plt.show()
    '''
    return CrfNerModel(tag_indexer, feature_indexer, optimizer.get_final_weights())


def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)