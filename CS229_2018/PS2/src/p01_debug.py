import matplotlib.pyplot as plt

import util

Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=False)

Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=False)

for i in range(len(Ya)):
    if Ya[i] == -1:
        Ya[i] = 0
        
for i in range(len(Yb)):
    if Yb[i] == -1:
        Yb[i] = 0

plt.figure()
util.plot_points(Xa, Ya)
plt.xlabel('x1')
plt.ylabel('y1')
plt.title('dataset A')
plt.savefig('output/p01_debug_A')

plt.figure()
util.plot_points(Xb, Yb)
plt.xlabel('x1')
plt.ylabel('y1')
plt.title('dataset B')
plt.savefig('output/p01_debug_B')

