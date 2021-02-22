import numpy as np
import matplotlib.pyplot as plt

time = [10, 50, 100, 500, 1000, 5000]
dos_acc = []
fuzzy_acc = []
for t in time:
    dos_res = np.load(f"results/DoS_{t}.npy", allow_pickle=True)
    dos_acc.append(dos_res[2][1])
    fuzzy_res = np.load(f"results/Fuzzy_{t}.npy", allow_pickle=True)
    fuzzy_acc.append(fuzzy_res[2][1])

print(dos_acc)
print(fuzzy_acc)

# ax = plt.gca()
# ax.set_ylim(0.0, 1.05)
# l1, = ax.plot([x for x in time], dos_acc, 'C0', marker='.')
# l2, = ax.plot([x for x in time], fuzzy_acc, 'C1', marker='.')
# ax.set_xlabel('unit observation time')
# ax.set_ylabel('f1 score')
# plt.legend([l1, l2], ["DoS", "Fuzzy"], loc='lower right')
# plt.grid(axis='y', which='major', linestyle='--')
# plt.savefig('f1.png')
# plt.show()