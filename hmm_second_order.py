import numpy as np
from collections import Counter, defaultdict
from data_loader import UNK_TOKEN, START_TOKEN, STOP_TOKEN

class SecondOrderHMM:
    """
    Second-Order (Trigram) Hidden Markov Model for Part-of-Speech Tagging.
    Implemented from scratch using Deleted Interpolation (Jelinek-Mercer / Brants TnT)
    for transition smoothing and fast 3D-vectorized Viterbi decoding.
    """
    def __init__(self, emission_k=1e-3, lambdas=None):
        """
        Args:
            emission_k (float): Laplace smoothing parameter for emissions.
            lambdas (tuple[float, float, float] | None): (lambda1, lambda2, lambda3) weights
                for linear interpolation. If None, computed automatically via Deleted Interpolation.
        """
        self.emission_k = emission_k
        self.lambdas = lambdas
        
        self.tags = []
        self.tag2idx = {}
        self.idx2tag = {}
        self.num_tags = 0
        
        self.vocab = set()
        self.log_start1 = None          # Shape: (num_tags,), log P(t1 | START, START)
        self.log_start2 = None          # Shape: (num_tags, num_tags), log P(t2 | START, t1)
        self.log_T = None               # Trigram transitions: (num_tags, num_tags, num_tags), log P(v | s, u)
        self.log_stop_trigram = None    # Shape: (num_tags, num_tags), log P(STOP | u, v)
        self.log_stop_start1 = None     # Shape: (num_tags,), log P(STOP | START, v) for length-1 sentences
        
        self.log_emissions = {}         # Map: word -> np.ndarray of shape (num_tags,)
        self.log_unk_emission = None    # Shape: (num_tags,)

    def _deleted_interpolation(self, trigram_counts, bigram_counts, unigram_counts, total_tokens):
        """
        Compute optimal linear interpolation weights (lambda1, lambda2, lambda3)
        using the Deleted Interpolation algorithm (Brants 2000).
        """
        l1, l2, l3 = 0.0, 0.0, 0.0
        
        for (t1, t2, t3), c123 in trigram_counts.items():
            if c123 == 0:
                continue
            
            c12 = bigram_counts.get((t1, t2), 0)
            c23 = bigram_counts.get((t2, t3), 0)
            c3 = unigram_counts.get(t3, 0)
            
            f3 = (c123 - 1.0) / (c12 - 1.0) if c12 > 1 else 0.0
            f2 = (c23 - 1.0) / (unigram_counts.get(t2, 0) - 1.0) if unigram_counts.get(t2, 0) > 1 else 0.0
            f1 = (c3 - 1.0) / (total_tokens - 1.0) if total_tokens > 1 else 0.0
            
            max_f = max(f1, f2, f3)
            if max_f > 0:
                if max_f == f3:
                    l3 += c123
                elif max_f == f2:
                    l2 += c123
                else:
                    l1 += c123
            else:
                l1 += c123
                
        total_l = l1 + l2 + l3
        if total_l == 0:
            return (0.1, 0.3, 0.6)
        return (l1 / total_l, l2 / total_l, l3 / total_l)

    def train(self, sentences, labels, vocab):
        """
        Train Second-Order HMM parameters.
        Args:
            sentences (list[list[str]]): Tokenized sentences.
            labels (list[list[str]]): Corresponding POS tags.
            vocab (set[str]): Predefined vocabulary.
        """
        self.vocab = set(vocab)
        unique_tags = sorted(list({tag for seq in labels for tag in seq}))
        self.tags = unique_tags
        self.num_tags = len(self.tags)
        self.tag2idx = {tag: i for i, tag in enumerate(self.tags)}
        self.idx2tag = {i: tag for i, tag in enumerate(self.tags)}
        K = self.num_tags
        
        # Counters for n-grams
        unigram_counts = Counter()
        bigram_counts = Counter()
        trigram_counts = Counter()
        
        # Word emissions per tag: word -> array(K)
        emission_counts = defaultdict(lambda: np.zeros(K, dtype=np.float64))
        tag_counts = np.zeros(K, dtype=np.float64)
        
        total_tokens = 0
        num_sentences = len(sentences)
        
        for sentence, label_seq in zip(sentences, labels):
            if not sentence or not label_seq:
                continue
            
            # Map sequence to indices and pad with START
            seq_indices = [self.tag2idx[t] for t in label_seq]
            padded_tags = [START_TOKEN, START_TOKEN] + seq_indices + [STOP_TOKEN]
            
            # Update n-gram counts
            for tag in seq_indices:
                unigram_counts[tag] += 1
                total_tokens += 1
                tag_counts[tag] += 1
                
            for i in range(len(padded_tags) - 1):
                bigram_counts[(padded_tags[i], padded_tags[i+1])] += 1
                
            for i in range(len(padded_tags) - 2):
                trigram_counts[(padded_tags[i], padded_tags[i+1], padded_tags[i+2])] += 1
                
            for word, tag_idx in zip(sentence, seq_indices):
                w = word if word in self.vocab else UNK_TOKEN
                emission_counts[w][tag_idx] += 1

        # Calculate or assign interpolation weights
        if self.lambdas is None:
            self.lambdas = self._deleted_interpolation(
                trigram_counts, bigram_counts, unigram_counts, total_tokens
            )
        l1, l2, l3 = self.lambdas
        
        # Unigram MLE distribution: P_mle(v)
        p_unigram = np.zeros(K, dtype=np.float64)
        for v in range(K):
            p_unigram[v] = unigram_counts[v] / total_tokens if total_tokens > 0 else 1.0 / K

        # 1. Start Transitions
        # P(t1 | START, START)
        start1_probs = np.zeros(K, dtype=np.float64)
        for v in range(K):
            c_tri = trigram_counts.get((START_TOKEN, START_TOKEN, v), 0)
            c_bi = bigram_counts.get((START_TOKEN, v), 0)
            
            p3 = c_tri / num_sentences if num_sentences > 0 else 0.0
            p2 = c_bi / num_sentences if num_sentences > 0 else 0.0
            p1 = p_unigram[v]
            start1_probs[v] = l3 * p3 + l2 * p2 + l1 * p1
        self.log_start1 = np.log(np.maximum(start1_probs, 1e-12))
        
        # P(t2 | START, t1) - shape: (K, K) for (t1, t2)
        start2_probs = np.zeros((K, K), dtype=np.float64)
        for u in range(K):
            c_start_u = bigram_counts.get((START_TOKEN, u), 0)
            c_u = unigram_counts[u]
            for v in range(K):
                c_tri = trigram_counts.get((START_TOKEN, u, v), 0)
                c_bi = bigram_counts.get((u, v), 0)
                
                p3 = c_tri / c_start_u if c_start_u > 0 else 0.0
                p2 = c_bi / c_u if c_u > 0 else 0.0
                p1 = p_unigram[v]
                start2_probs[u, v] = l3 * p3 + l2 * p2 + l1 * p1
        self.log_start2 = np.log(np.maximum(start2_probs, 1e-12))

        # 2. General Trigram Transition Matrix: P(v | s, u) - shape: (K, K, K)
        T_probs = np.zeros((K, K, K), dtype=np.float64)
        for s in range(K):
            for u in range(K):
                c_su = bigram_counts.get((s, u), 0)
                c_u = unigram_counts[u]
                for v in range(K):
                    c_tri = trigram_counts.get((s, u, v), 0)
                    c_bi = bigram_counts.get((u, v), 0)
                    
                    p3 = c_tri / c_su if c_su > 0 else 0.0
                    p2 = c_bi / c_u if c_u > 0 else 0.0
                    p1 = p_unigram[v]
                    T_probs[s, u, v] = l3 * p3 + l2 * p2 + l1 * p1
        self.log_T = np.log(np.maximum(T_probs, 1e-12))
        
        # 3. Stop Transitions
        # P(STOP | u, v) - shape: (K, K)
        stop_probs = np.zeros((K, K), dtype=np.float64)
        p_stop_uni = num_sentences / (total_tokens + num_sentences)
        for u in range(K):
            c_u = unigram_counts[u]
            for v in range(K):
                c_uv = bigram_counts.get((u, v), 0)
                c_tri = trigram_counts.get((u, v, STOP_TOKEN), 0)
                c_bi = bigram_counts.get((v, STOP_TOKEN), 0)
                c_v = unigram_counts[v]
                
                p3 = c_tri / c_uv if c_uv > 0 else 0.0
                p2 = c_bi / c_v if c_v > 0 else 0.0
                p1 = p_stop_uni
                stop_probs[u, v] = l3 * p3 + l2 * p2 + l1 * p1
        self.log_stop_trigram = np.log(np.maximum(stop_probs, 1e-12))
        
        # P(STOP | START, v) for length-1 sentence - shape: (K,)
        stop_start1_probs = np.zeros(K, dtype=np.float64)
        for v in range(K):
            c_start_v = bigram_counts.get((START_TOKEN, v), 0)
            c_tri = trigram_counts.get((START_TOKEN, v, STOP_TOKEN), 0)
            c_bi = bigram_counts.get((v, STOP_TOKEN), 0)
            c_v = unigram_counts[v]
            
            p3 = c_tri / c_start_v if c_start_v > 0 else 0.0
            p2 = c_bi / c_v if c_v > 0 else 0.0
            p1 = p_stop_uni
            stop_start1_probs[v] = l3 * p3 + l2 * p2 + l1 * p1
        self.log_stop_start1 = np.log(np.maximum(stop_start1_probs, 1e-12))

        # 4. Emissions with Add-k smoothing
        denom_emiss = tag_counts + self.emission_k * len(self.vocab)
        unk_counts = emission_counts[UNK_TOKEN]
        unk_probs = (unk_counts + self.emission_k) / denom_emiss
        self.log_unk_emission = np.log(unk_probs)
        
        self.log_emissions = {}
        for word in self.vocab:
            w_counts = emission_counts[word]
            probs = (w_counts + self.emission_k) / denom_emiss
            self.log_emissions[word] = np.log(probs)

    def _get_emission_log_prob(self, word):
        """Retrieve log emission vector for a word (or UNK fallback)."""
        return self.log_emissions.get(word, self.log_unk_emission)

    def viterbi(self, sentence):
        """
        Find the most probable POS tag sequence using second-order Viterbi decoding.
        Args:
            sentence (list[str]): List of words.
        Returns:
            list[str]: Predicted sequence of POS tags.
        """
        N = len(sentence)
        if N == 0:
            return []
        
        K = self.num_tags
        
        # Special case: Sentence of length 1
        if N == 1:
            emiss_0 = self._get_emission_log_prob(sentence[0])
            scores = self.log_start1 + emiss_0 + self.log_stop_start1
            best_tag_idx = int(np.argmax(scores))
            return [self.idx2tag[best_tag_idx]]
            
        # Special case: Sentence of length 2
        emiss_0 = self._get_emission_log_prob(sentence[0])
        V0 = self.log_start1 + emiss_0  # shape: (K,)
        
        emiss_1 = self._get_emission_log_prob(sentence[1])
        # V1[u, v] = V0[u] + log_start2[u, v] + emiss_1[v]
        V1 = V0[:, None] + self.log_start2 + emiss_1[None, :]  # shape: (K, K)
        
        if N == 2:
            scores = V1 + self.log_stop_trigram
            best_idx = np.unravel_index(np.argmax(scores), scores.shape)
            return [self.idx2tag[best_idx[0]], self.idx2tag[best_idx[1]]]
            
        # General case: N >= 3
        # V_table[t] will store (K, K) table for step t
        # BP_table[t] stores argmax_s of shape (K, K) for step t
        V_prev = V1  # shape: (K, K) at t=1
        BP_list = [None, None]  # placeholders for t=0, t=1
        
        for t in range(2, N):
            emiss_t = self._get_emission_log_prob(sentence[t])  # shape: (K,)
            # M[s, u, v] = V_prev[s, u] + log_T[s, u, v]
            M = V_prev[:, :, None] + self.log_T  # shape: (K, K, K)
            
            # Maximize over s (axis 0)
            V_curr = np.max(M, axis=0) + emiss_t[None, :]  # shape: (K, K)
            BP_curr = np.argmax(M, axis=0)  # shape: (K, K)
            
            BP_list.append(BP_curr)
            V_prev = V_curr

        # Termination: step N - 1 with STOP transition
        final_scores = V_prev + self.log_stop_trigram  # shape: (K, K)
        best_u, best_v = np.unravel_index(np.argmax(final_scores), final_scores.shape)
        
        # Backtracking
        best_path_indices = [0] * N
        best_path_indices[-1] = int(best_v)
        best_path_indices[-2] = int(best_u)
        
        curr_u, curr_v = int(best_u), int(best_v)
        for t in range(N - 1, 1, -1):
            s = BP_list[t][curr_u, curr_v]
            best_path_indices[t - 2] = int(s)
            curr_v = curr_u
            curr_u = int(s)
            
        return [self.idx2tag[idx] for idx in best_path_indices]

    def predict(self, sentences):
        """Tag a collection of sentences."""
        return [self.viterbi(s) for s in sentences]
