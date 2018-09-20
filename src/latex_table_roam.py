import pickle
import sys

with open(sys.argv[1], 'rb') as f:
    table = pickle.load(f)

for m in table:
    for i in m[1:]:
        print(i, end=' & ')
    print('\\\\')