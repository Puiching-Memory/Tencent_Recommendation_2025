import pickle

with open('data/TencentGR_1k/seq_offsets.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)