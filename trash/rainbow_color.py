import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pdb

NUM_COLORS = 20

# cm = plt.get_cmap('gist_rainbow')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
# pdb.set_trace()
# for i in range(NUM_COLORS):
#     ax.plot(np.arange(10)*(i+1))
#
# fig.savefig('moreColors.png')
# plt.show()

cmap_names = ['tab20']
for cmap_name in cmap_names:
    cm = plt.cm.get_cmap(cmap_name)
    f, ax = plt.subplots(1, 1)
    for idx, rgb in enumerate(cm.colors):
        print(rgb)
        x = idx % 10
        y = -int(idx / 10)
        ax.scatter(x, y, s=10**2, color=rgb)
plt.show()