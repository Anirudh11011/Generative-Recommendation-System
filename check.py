import pickle
d = pickle.load(open("data/user_sequences.pkl", "rb"))

k = next(iter(d))
v = d[k]
print("num users:", len(d))
print("key:", k)
print("value type:", type(v))
print("value:", v)