import numpy as np
from collections import Counter, defaultdict
from data_loader import UNK_TOKEN, START_TOKEN, STOP_TOKEN

class FirstOrderHMM:
    """
    First-Order Hidden Markov Model (HMM) for Part-of-Speech Tagging.
    Implemented from scratch using maximum likelihood estimation with Laplace smoothing
    and fast vectorized Viterbi decoding.
    """
    def __init__(self, emission_k=1e-3, transition_k=1e-3):
        self.emission_k = emission_k
        self.transition_k = transition_k
        
        self.tags = []
        self.tag2idx = {}
        self.idx2tag = {}
        self.num_tags = 0
        
        self.vocab = set()
        self.log_pi = None       # Shape: (num_tags,)
        self.log_A = None        # Transition matrix: (num_tags, num_tags), log P(tag_j | tag_i)
        self.log_stop = None     # Shape: (num_tags,), log P(STOP | tag_i)
        self.log_emissions = {}  # Map: word -> np.ndarray of shape (num_tags,)
        self.log_unk_emission = None # Shape: (num_tags,)

    def train(self, sentences, labels, vocab):
        """
        Train First-Order HMM parameters from annotated training corpus.
        Args:
            sentences (list[list[str]]): Tokenized sentences.
            labels (list[list[str]]): Tag sequences matching sentences.
            vocab (set[str]): Predefined vocabulary (including UNK_TOKEN).
        """
        self.vocab = set(vocab)
        
        # Collect unique tags
        unique_tags = sorted(list({tag for seq in labels for tag in seq}))
        self.tags = unique_tags
        self.num_tags = len(self.tags)
        self.tag2idx = {tag: i for i, tag in enumerate(self.tags)}
        self.idx2tag = {i: tag for i, tag in enumerate(self.tags)}
        
        # Initialize count structures
        start_counts = np.zeros(self.num_tags, dtype=np.float64)
        transition_counts = np.zeros((self.num_tags, self.num_tags), dtype=np.float64)
        stop_counts = np.zeros(self.num_tags, dtype=np.float64)
        tag_counts = np.zeros(self.num_tags, dtype=np.float64)
        emission_counts = defaultdict(lambda: np.zeros(self.num_tags, dtype=np.float64))
        
        num_sentences = len(sentences)
        vocab_size = len(self.vocab)
        
        # Count occurrences
        for sentence, label_seq in zip(sentences, labels):
            if not sentence or not label_seq:
                continue
            
            # Start transition
            first_tag_idx = self.tag2idx[label_seq[0]]
            start_counts[first_tag_idx] += 1
            
            for i, (word, tag) in enumerate(zip(sentence, label_seq)):
                tag_idx = self.tag2idx[tag]
                tag_counts[tag_idx] += 1
                
                # Replace OOV words with UNK_TOKEN during counting
                w = word if word in self.vocab else UNK_TOKEN
                emission_counts[w][tag_idx] += 1
                
                # Transitions
                if i > 0:
                    prev_tag_idx = self.tag2idx[label_seq[i - 1]]
                    transition_counts[prev_tag_idx, tag_idx] += 1
            
            # Stop transition
            last_tag_idx = self.tag2idx[label_seq[-1]]
            stop_counts[last_tag_idx] += 1

        # 1. Compute Start Probabilities with Add-k smoothing
        pi_probs = (start_counts + self.transition_k) / (num_sentences + self.transition_k * self.num_tags)
        self.log_pi = np.log(pi_probs)
        
        # 2. Compute Transition Probabilities with Add-k smoothing
        # Total transitions from each tag: transitions to other tags + stop transition
        denom_trans = transition_counts.sum(axis=1) + stop_counts + self.transition_k * (self.num_tags + 1)
        A_probs = (transition_counts + self.transition_k) / denom_trans[:, None]
        stop_probs = (stop_counts + self.transition_k) / denom_trans
        self.log_A = np.log(A_probs)
        self.log_stop = np.log(stop_probs)
        
        # 3. Compute Emission Probabilities with Add-k smoothing
        # P(word | tag) = (count(tag, word) + k) / (count(tag) + k * |V|)
        denom_emiss = tag_counts + self.emission_k * vocab_size
        
        # Default emission for UNK (and fallback for unseen words)
        unk_counts = emission_counts[UNK_TOKEN]
        unk_probs = (unk_counts + self.emission_k) / denom_emiss
        self.log_unk_emission = np.log(unk_probs)
        
        self.log_emissions = {}
        for word in self.vocab:
            w_counts = emission_counts[word]
            probs = (w_counts + self.emission_k) / denom_emiss
            self.log_emissions[word] = np.log(probs)

    def _get_emission_log_prob(self, word):
        """Retrieve pre-computed log emission vector for a word (or UNK fallback)."""
        return self.log_emissions.get(word, self.log_unk_emission)

    def viterbi(self, sentence):
        """
        Find the most probable sequence of POS tags for a given sentence.
        Args:
            sentence (list[str]): List of words.
        Returns:
            list[str]: Predicted sequence of POS tags.
        """
        N = len(sentence)
        if N == 0:
            return []
        
        # V[tag_idx, t] stores the highest log-probability of a path ending at tag_idx at step t
        V = np.zeros((self.num_tags, N), dtype=np.float64)
        BP = np.zeros((self.num_tags, N), dtype=np.int32)
        
        # Initial step (t = 0)
        emiss_0 = self._get_emission_log_prob(sentence[0])
        V[:, 0] = self.log_pi + emiss_0
        
        # Recursion step (t = 1 ... N - 1)
        for t in range(1, N):
            emiss_t = self._get_emission_log_prob(sentence[t])
            # all_scores[u, v] = V[u, t - 1] + log_A[u, v]
            all_scores = V[:, t - 1, None] + self.log_A
            V[:, t] = np.max(all_scores, axis=0) + emiss_t
            BP[:, t] = np.argmax(all_scores, axis=0)
        
        # Termination step with STOP transition
        final_scores = V[:, N - 1] + self.log_stop
        best_last_tag = int(np.argmax(final_scores))
        
        # Backtracking
        best_path_indices = [0] * N
        best_path_indices[-1] = best_last_tag
        for t in range(N - 1, 0, -1):
            best_path_indices[t - 1] = BP[best_path_indices[t], t]
            
        return [self.idx2tag[idx] for idx in best_path_indices]

    def predict(self, sentences):
        """Tag a collection of sentences."""
        return [self.viterbi(s) for s in sentences]
