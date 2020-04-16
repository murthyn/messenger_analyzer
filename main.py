from sentence_transformers import SentenceTransformer
import numpy as np
import re, os, glob

def load_data(filenames=['Ayush-Sharma.txt', 'Debaditya-Pramanik.txt', 'Gabriella-Garcia.txt', 'Nikhil-Murthy.txt']):
    """
    Loads Messenger data

    Args:
        filenames (list of strs): paths to txt files where messenger chats are stored (one file per person)
            filename must be in format "FirstName-LastName.txt"

    Returns:
        sentence_dict (dict: file_name -> list of sentences)
    """ 

    sentence_dict = {}

    for file_name in filenames:
        try:
            file = open(os.path.join("data", file_name), "r")
            sentence_dict[file_name] = file.readlines()
        except:
            print(f"Unable to open {file}")

    return sentence_dict

def save_encodings(filename, sentences):
    """
    Saves Numpy array containing sentence emebeddings

    Args:
        filename (str): name of file where embeddings will be saved
        sentences (list of str): list of sentences to be encoded
    
    Returns:
        None
    """
    # download pretrained BERT model
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    encodings = model.encode(sentences)
    np.save(os.path.join("data", filename), encodings)

def save_all_encodings(sentence_dict):
    """
    Saves Numpy arrays containing sentence emebeddings for all people

    Args:
        sentence_dict (dict: file_name -> list of sentences)
    
    Returns:
        None
    """
    for file_name in sentence_dict:
        new_file_name = file_name.strip(".txt")
        save_encodings(new_file_name, sentence_dict[file_name])


def load_all_encodings():
    """
    Load all encodings in data directory (searches for all numpy arrays and loads)
    
    Args:
        None

    Returns:
        encodings_dict (dict: file_name -> np array of encoding)
    """

    encodings_dict = {}

    for file in os.listdir("data"):
        if file.endswith(".npy"):
            path = os.path.join("data", file)
            encodings_dict[path] = np.load(path)

    return encodings_dict



# sentence_dict = load_data()
# save_all_encodings(sentence_dict)

encodings_dict = load_all_encodings()
print([encodings_dict[name].shape for name in encodings_dict])





