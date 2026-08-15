import os
import json
from collections import Counter
import kagglehub

UNK_TOKEN = "<UNK>"
START_TOKEN = "<START>"
STOP_TOKEN = "<STOP>"

def download_dataset():
    """Download the POS tagging dataset from KaggleHub and return directory path."""
    dataset_path = kagglehub.dataset_download("pranav13300/annotated-dataset-for-pos-tagging")
    return dataset_path

def load_data_split(file_path):
    """
    Load sentences and labels from a json file.
    Returns:
        sentences (list[list[str]]): List of sentences (tokens).
        labels (list[list[str]]): List of label sequences.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    sentences = []
    labels = []
    for item in data:
        sentences.append(item["sentence"])
        labels.append(item["labels"])
    return sentences, labels

def load_dataset(data_dir=None):
    """
    Download and load both train and dev sets.
    Returns:
        (train_sentences, train_labels), (dev_sentences, dev_labels)
    """
    if data_dir is None:
        data_dir = download_dataset()
    
    train_path = os.path.join(data_dir, "train.json")
    dev_path = os.path.join(data_dir, "dev.json")
    
    train_sentences, train_labels = load_data_split(train_path)
    dev_sentences, dev_labels = load_data_split(dev_path)
    
    return (train_sentences, train_labels), (dev_sentences, dev_labels)

def build_vocabulary(train_sentences, min_freq=2, unk_token=UNK_TOKEN):
    """
    Build vocabulary from training sentences.
    Words with frequency < min_freq are mapped to unk_token.
    Returns:
        vocab (set[str]): Set of recognized vocabulary words including unk_token.
        word_counts (Counter): Raw counts of words in the training set.
    """
    word_counts = Counter(word for sentence in train_sentences for word in sentence)
    vocab = {word for word, count in word_counts.items() if count >= min_freq}
    vocab.add(unk_token)
    return vocab, word_counts

def preprocess_sentence(sentence, vocab, unk_token=UNK_TOKEN):
    """Replace out-of-vocabulary tokens in a sentence with unk_token."""
    return [word if word in vocab else unk_token for word in sentence]

def preprocess_dataset(sentences, vocab, unk_token=UNK_TOKEN):
    """Preprocess an entire list of sentences with vocabulary filtering."""
    return [preprocess_sentence(s, vocab, unk_token) for s in sentences]
