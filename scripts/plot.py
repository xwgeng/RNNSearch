import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


with open('../loss', 'r') as f:
    y1 = []
    y2 = []
    for line in f:
        line = line.strip().split()
        if len(line) < 2:
            continue
        try:
            line = map(lambda x: float(x), line)
        except:
            pass
        else:
            y1.append(line[0])
            y2.append(line[1])
plt.suptitle('Loss Curves [lr=5e-4]')
plt.subplot(211)
plt.plot(y1)
plt.xlabel('iter')
plt.ylabel('loss per sentence')
plt.subplot(212)
plt.plot(y2)
plt.xlabel('iter')
plt.ylabel('loss per word')
plt.savefig('Loss-lr_0.001.png')
