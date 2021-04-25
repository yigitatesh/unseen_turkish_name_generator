print("Loading Packages...")

import os

# Do not show unnecessary warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# USE FULL POWER OF GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
import numpy as np

### Load Data
print("Loading Data...")

with open("data/names_ascii.txt") as f:
    names = f.readlines()

# create tr to ascii dict
tr2ascii_dict = {"ç":"c", "ğ":"g", "ı":"i", "ö":"o", "ş":"s", "ü":"u"}

### Prepare Data

#### Tokenizer

# create tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
)

# fit in data
tokenizer.fit_on_texts("".join(names))

# mapping dictionaries between indexes and characters
char_to_index = tokenizer.word_index
index_to_char = dict((v, k) for k, v in char_to_index.items())

#### Seq to Name and Name to Seq

def name_to_seq(name):
    return [tokenizer.texts_to_sequences(ch)[0][0] for ch in name]

def seq_to_name(seq):
    return "".join([index_to_char[i] for i in seq])

#### Get max len of names and number of possible characters

# find max len
max_len = max([len(name) for name in names])

# find number of characters
num_chars = len(char_to_index) + 1 # +1 is null character(0)

### Load the model
print("Loading the Generator...")
model = tf.keras.models.load_model("model/tr_name_generate_model.h5")

### Generate Names

## Helper Functions

def is_real_name(name):
    """checks whether created name is in names dataset or not"""
    for real_name in names:
        if name == real_name:
            return True
    return False

def tr2ascii(name):
    newname = ""
    for char in name:
        if char in tr2ascii_dict:
            newname += tr2ascii_dict[char]
        else:
            newname += char
    return newname

## Generate Name Function

def generate_names(seed=""):
    """Gets a seed (char or string)
    
    In a loop, predicts next chars and appends them to seed
    Next chars are chosen using probabilities to create different names each time
    checks created name whether it is a real name or not
    if it is a real name creates another name
    
    Returns created name"""

    # keep raw seed in memory
    seed = tr2ascii(seed)
    raw_seed = seed
    
    for i in range(40):
        # prepare sequence
        seq = name_to_seq(seed)
        padded = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_len-1,
                                                               padding="pre", 
                                                               truncating="pre")
        
        # predict next char
        probs = model.predict(padded)[0]
        index = np.random.choice(list(range(num_chars)), p=probs.ravel())
        
        # null character
        if index == 0:
            break

        pred_char = index_to_char[index]
        seed += pred_char
        
        # end of name
        if pred_char == "\n":
            break

    # if name is real, create another name
    if is_real_name(seed):
        seed = generate_names(seed=raw_seed)

    return seed

# menu function
def menu():
    print("\nType '1' to generate a turkish name.")
    print("Type '2' to generate a turkish name starting with characters you will enter.")
    print("Type '3' to generate turkish names many as numbers you will enter.")
    print("Type '4' to generate turkish names starting with characters you will enter.")
    print("Type '5' to exit.")


# main function
def main():
    # dummy prediction for initialization
    model.predict(np.zeros((1, max_len-1)))

    print("\nWelcome to the Turkish Name Generator!")
    print("These created names will NOT be REAL NAMES!")
    print("They are being created by Artificial Intelligence.")
    
    run = True
    while run:
        # print menu
        menu()

        # get input
        choice = input("\nType your choice here: ")

        # process choice
        if choice == "1":
            name = generate_names(seed="")
            print("\nYour Turkish Name: {}".format(name))

        elif choice == "2":
            # get initial characters
            valid = False
            while not valid:
                seed = input("\nType initial characters: ")

                if seed.isalpha():
                    seed = seed.lower()
                    valid = True
                else:
                    print("\nPlease type alphabetical character(s).")

            # generate name
            name = generate_names(seed=seed)
            print("\nYour Turkish Name: {}".format(name))

        elif choice == "3":
            # get number of names
            valid = False
            while not valid:
                num_names = input("\nType number of Turkish names: ")

                try:
                    num_names = int(num_names)
                    valid = True
                except:
                    print("\nPlease type an integer number.")

            # generate names
            print("\nYour Turkish Names:\n")

            for i in range(num_names):
                name = generate_names(seed="")
                print("{}: {}".format(i+1, name))

        elif choice == "4":
            # get number of names
            valid = False
            while not valid:
                num_names = input("\nType number of Turkish names: ")

                try:
                    num_names = int(num_names)
                    valid = True
                except:
                    print("\nPlease type an integer number.")

            # get initial characters
            valid = False
            while not valid:
                seed = input("\nType initial characters: ")

                if seed == "":
                    valid = True
                elif seed.isalpha():
                    seed = seed.lower()
                    valid = True
                else:
                    print("\nPlease type alphabetical character(s).")

            # generate names
            print("\nYour Turkish Names starting with '{}':\n".format(seed))

            for i in range(num_names):
                name = generate_names(seed=seed)
                print("{}: {}".format(i+1, name))

        elif choice == "5":
            run = False

        else:
            print("\nNot a valid choice!")


# START THE GENERATOR APP
main()
