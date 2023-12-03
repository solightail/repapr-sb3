import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("Pendulum-v1", render_mode='human')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, progress_bar=True)

vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    #if done:
        #obs = env.reset()

env.close()