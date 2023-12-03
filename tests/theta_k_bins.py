import numpy as np

action_list = [-1, 0, 1]
tones = 6

action_arr = np.array([action_list] * tones)
empty = np.empty(tones)
flinf = float("inf")
npinf = np.inf

indexes: np.ndarray = np.arange(tones)
theta_ks: np.ndarray = ((indexes) * (indexes - 1)) / (2*(tones - 1))
theta_ks_bins: np.ndarray = theta_ks % 1
theta_k_bins: np.ndarray = (((indexes)*(indexes - 1)) / (2*(tones - 1))) % 1


print(action_arr)
print(empty)
print(flinf)
print(npinf)

print(theta_ks)
print(theta_ks_bins)
print(theta_k_bins)