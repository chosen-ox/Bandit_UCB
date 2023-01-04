import sys
import matplotlib.pyplot as plt
import numpy as np

# filename = sys.argv[1]
f = open('D:/Academic/RateControl/Bandit_UCB/datas/ucb_1', 'r')
for i in range(6):
    f.readline()
rate_list = []
line = f.readline().split()
while len(line) != 0:
    try:
        rate = float(line[6])
        if line[7] == "Kbits/sec":
            print(rate)
            rate >>= 3
    except:
        line = f.readline().split()
        continue
    rate_list.append(rate)
    line = f.readline().split()
print(rate_list)
y = np.array(rate_list)
# x = np.arange(554)
plt.plot( y, 'ro')
plt.show()