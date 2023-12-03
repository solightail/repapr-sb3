gamma = 0.999

# gamma 整数表示 + ゼロ埋め
pG = int(gamma * (10 ** len(str(gamma).split(".")[1])))
if gamma < 1: pG = str(pG).zfill(len(str(gamma).split(".")[1]) + 1)