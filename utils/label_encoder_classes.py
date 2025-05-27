import pickle

file_path = '/path/to/label_encoder.pkl'

with open(file_path, 'rb') as file:
    label_encoder = pickle.load(file)

print("Labels:", label_encoder.classes_)









