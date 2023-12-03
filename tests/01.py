import numpy as np

# 長さ4の np.ndarray を2つ作る
a = np.array([0.2, 0.8, 0.9, 0.2])
b = np.array([0.5, 0.4, 0.9, -0.5])

# 要素ごとの和を計算する
c = np.add(a, b) # または c = a + b
e = np.mod(c, 1)

# 条件分岐を使って 0~1 の範囲に収める
d = np.zeros(4) # 長さ4のゼロ配列を作る
for i in range(4): # 配列の各要素に対してループする
    if c[i] <= 0: # 値が0以下なら
        d[i] = c[i] + 1 # 1を足す
    else: # そうでないなら
        d[i] = np.mod(c[i], 1) # 剰余演算を行う

# 結果を表示する
print(d)
