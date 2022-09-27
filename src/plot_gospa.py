import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as st


total_gospa = np.load('task4/total_gospa.npy')
total_loc = np.load('task4/total_loc.npy')
total_miss = np.load('task4/total_miss.npy')
total_false = np.load('task4/total_false.npy')

print("======== 95% students-t ========")
print(f"GOSPA: {total_gospa.mean()} +/- {max(st.t.interval(0.95, len(total_gospa)-1, loc=0, scale=st.sem(total_gospa)))}")
print(f"LOC: {total_loc.mean()} +/- {max(st.t.interval(0.95, len(total_loc)-1, loc=0, scale=st.sem(total_loc)))}")
print(f"MISS: {total_miss.mean()} +/- {max(st.t.interval(0.95, len(total_miss)-1, loc=0, scale=st.sem(total_miss)))}")
print(f"FALSE: {total_false.mean()} +/- {max(st.t.interval(0.95, len(total_false)-1, loc=0, scale=st.sem(total_false)))}")


print("======== 95% normal ========")
print(f"GOSPA: {total_gospa.mean():.2f} +/- {max(st.norm.interval(0.95, loc=0, scale=st.sem(total_gospa))):.2f}")
print(f"LOC: {total_loc.mean():.2f} +/- {max(st.norm.interval(0.95, loc=0, scale=st.sem(total_loc))):.2f}")
print(f"MISS: {total_miss.mean():.2f} +/- {max(st.norm.interval(0.95, loc=0, scale=st.sem(total_miss))):.2f}")
print(f"FALSE: {total_false.mean():.2f} +/- {max(st.norm.interval(0.95, loc=0, scale=st.sem(total_false))):.2f}")


plt.hist(total_gospa, bins = 50)
plt.xlabel("Total GOSPA")
plt.ylabel("N")
plt.show()