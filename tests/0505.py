import numpy as np

# 数値の配列を作成する
values = np.array([1.5, -1.6, 0.3, 0.7, -0.2, -0.6, 0.9, -0.8])

# 0~1の範囲にする
values = np.mod(values, 1)

# 0.5より大きい場合は、-1を足す
values = np.where(values > 0.5, values - 1, values)

# -0.5より小さい場合は、1を足す
values = np.where(values < -0.5, values + 1, values)

# 結果を表示する
print(values)