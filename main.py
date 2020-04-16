from sentence_transformers import SentenceTransformer
import numpy as np
import re, os, glob
import torch

def load_data(filenames=['Ayush-Sharma.txt', 'Debaditya-Pramanik.txt', 'Gabriella-Garcia.txt', 'Nikhil-Murthy.txt', 'Keshav-Gupta.txt']):
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

def create_nn_data(encodings_dict):
    """
    Create PyTorch tensor X (embedings) and Y (targets) for training

    Args:
        encodings_dict (dict: file_name -> np array of encoding)

    Returns:
        X_shuffled (torch.Tensor): (num_encodings, encoding)
        Y_shuffled (torch.Tensor) (target_user)
        target_index_to_name (dict: int -> str): dict mapping target index to name
    """

    # num_encodings = sum([encodings_dict[name].shape[0] for name in encodings_dict])
    min_num_encodings = min([encodings_dict[name].shape[0] for name in encodings_dict])
    num_encodings = min_num_encodings * len(encodings_dict.keys())
    encoding_size = [encodings_dict[name].shape[1] for name in encodings_dict][0]

    X = torch.ones((num_encodings, encoding_size))
    Y = torch.ones((num_encodings))

    names = list(encodings_dict.keys())
    target_index_to_name = {names[i] : i for i in range(len(names))}

    i = 0 # first unfilled index in X

    for name in encodings_dict:
        encodings = encodings_dict[name][np.random.choice(encodings_dict[name].shape[0], size=min_num_encodings, replace=False)]
        X[i:i+min_num_encodings] = torch.from_numpy(encodings)
        Y[i:i+min_num_encodings] = target_index_to_name[name]
        i += encodings.shape[0]

    # shuffle X and Y
    shuffle_order = np.array(range(num_encodings))
    np.random.shuffle(shuffle_order) 
    X_shuffled = X[shuffle_order]
    Y_shuffled = Y[shuffle_order]

    # convert Y_shuffled to long
    Y_shuffled = Y_shuffled.type(torch.LongTensor)

    return X_shuffled, Y_shuffled, target_index_to_name



# sentence_dict = load_data()
# save_all_encodings(sentence_dict)



encodings_dict = load_all_encodings()
x, y, target_index_to_name = create_nn_data(encodings_dict)

print(target_index_to_name)

encoding_size = x.shape[1]
num_names = len(encodings_dict.keys())
hidden_size = 128

num_train = int(x.shape[0] * 0.8)
x_train, y_train = x[:num_train], y[:num_train]
x_test, y_test = x[num_train:], y[num_train:]

model = torch.nn.Sequential(
    torch.nn.Linear(encoding_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, num_names),
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

def train_model(model):
    for t in range(1000):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)   

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 99:
            print(t, loss.item())

            # Calculate train and test accuracies
            train_accuracy = accuracy(y_pred, y_train)
            y_pred = model(x_test)
            test_accuracy = accuracy(y_pred, y_test)
            print("Train accuracy", train_accuracy)
            print("Test accuracy", test_accuracy)

            print("0:", torch.sum(0 == torch.argmax(torch.nn.functional.softmax(y_pred), 1)).item()/y_pred.shape[0])
            print("1:", torch.sum(1 == torch.argmax(torch.nn.functional.softmax(y_pred), 1)).item()/y_pred.shape[0])
            print("2:", torch.sum(2 == torch.argmax(torch.nn.functional.softmax(y_pred), 1)).item()/y_pred.shape[0])
            print("3:", torch.sum(3 == torch.argmax(torch.nn.functional.softmax(y_pred), 1)).item()/y_pred.shape[0])
            print("4:", torch.sum(4 == torch.argmax(torch.nn.functional.softmax(y_pred), 1)).item()/y_pred.shape[0])

    return model

def accuracy(y_pred, y):
    return torch.sum(y == torch.argmax(torch.nn.functional.softmax(y_pred), 1)).item() / y.shape[0]

# model = train_model(model)
# torch.save(model.state_dict(), "model.pt")


model = torch.nn.Sequential(
    torch.nn.Linear(encoding_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, num_names),
)
model.load_state_dict(torch.load("model.pt"))
model.eval()

    
def test_sentences(model, sentences):
    nlp_model = SentenceTransformer('bert-base-nli-mean-tokens')
    encodings = nlp_model.encode(sentences)

    x = torch.from_numpy(np.stack(encodings))
    y = model(x)
    classes = torch.argmax(torch.nn.functional.softmax(y), 1)

    name_to_target_index = {target_index_to_name[k] : k for k in target_index_to_name}

    sentence_dict = {sentences[i] : name_to_target_index[int(classes[i])].strip("data/").strip(".npy") for i in range(len(sentences))}

    return sentence_dict

print(test_sentences(model, ["I like to drink", "Liberals and the left wing media are politicially correct and hate on Trump for no reason", "hahahaha", "I have cricket practice"]))




