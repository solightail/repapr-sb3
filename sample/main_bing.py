import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 環境の作成
env = gym.make('Pendulum-v1', render_mode='human')

# モデルの作成
model = PPO("MlpPolicy", env, verbose=1)

# 学習の実行
model.learn(total_timesteps=10000)

# 学習したモデルの保存
model.save("ppo_pendulum")

# 保存したモデルの読み込み
model = PPO.load("ppo_pendulum")

# テストランの実行
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
