import os
import json
from random import shuffle
from sklearn.model_selection import train_test_split


main_directory = os.path.dirname(os.getcwd())
    
# Folder z dabtnu
metadata_directory = os.path.join(main_directory, "data", "AudioMNIST", "data","audioMNIST_meta.json")
# Wczytanie metadanych
with open(metadata_directory, 'r') as f:
    metadata = json.load(f)

train_files = []
test_files = []

# Z każdego speakera oddzielamy train i test
for speaker_id in metadata.keys():
    for digit in range(10):
        # bieżemy wszystkie 50 nagrań speakera
        digit_files = [f"{speaker_id}_{digit}_{rep}.wav" for rep in range(50)]
        
        # train-test split
        digit_train, digit_test = train_test_split(
            digit_files, test_size=0.2, random_state=0, shuffle=True
        )
        
        train_files.extend(digit_train)
        test_files.extend(digit_test)
shuffle(train_files)
shuffle(test_files)
print(train_files, "\n")
print("# Test:", test_files)