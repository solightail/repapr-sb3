import os
import gymnasium as gym

from .modules.conf import Conf

cfg = Conf()
out = f"{cfg.filepath}/{cfg.algorithm}/N{cfg.tones}/{cfg.eval_model}_LT{cfg.total_timesteps}_LE{cfg.max_episode_steps}"
# unuse names
# _G{str(cfg.gamma).replace('.', '')}
env = gym.make("repapr-v0", render_mode="debug")

# 学習アルゴリズム選択
match cfg.algorithm:
    case "PPO":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1, gamma=cfg.gamma)
    case "SAC":
        from stable_baselines3 import SAC
        model = SAC("MlpPolicy", env, verbose=1, gamma=cfg.gamma)
    case _:
        raise ValueError("A non-existent algorithm is selected.")


def program() -> None:
    match cfg.mode:
        case 0:
            learn()
            exec()
        case 1:
            learn()
        case 2:
            exec()
        case _:
            raise ValueError("A non-existent mode is selected.")

def learn() -> None:
    # 学習モデルの上書き確認
    if os.path.exists(f"{out}.zip"):
        if not input('Overwrite learned files? [Y/n] ') in ["Y", "y", "YES", "Yes", "yes"]:
            raise FileExistsError("Learned file already exists.")

    if cfg.notify is True: _notify(f"学習を開始します")
    match cfg.algorithm:
        case "PPO":
            model.learn(total_timesteps=cfg.total_timesteps, progress_bar=True)
        case "SAC":
            model.learn(total_timesteps=cfg.total_timesteps, log_interval=4, progress_bar=True)
        case _:
            raise ValueError("A non-existent algorithm is selected.")
    model.save(out)
    if cfg.notify is True: _notify(f"学習が完了しました")

def _notify(in_msg) -> None:
    from datetime import datetime
    from .modules.utils import send_line
    end = datetime.now()
    message = f"repapr-ppo-cas / {end.time().isoformat(timespec='seconds')}\n{cfg.algorithm} N{cfg.tones}_LT{cfg.total_timesteps}_LE{cfg.max_episode_steps}\n{in_msg}"
    send_line(cfg.line['channel_token'], cfg.line['user_id'], message)


def exec() -> None:
    # 学習モデルの存在確認
    if not os.path.exists(f"{out}.zip"):
        raise FileNotFoundError("Learned file not found.")

    # 変数初期化
    exec_loop = 5
    exec_limit = cfg.max_episode_steps*exec_loop
    list_step = [i for i in range(exec_limit)]
    list_epi = [i for i in range(exec_loop) for _ in range(cfg.max_episode_steps)]
    list_action, list_theta_k_bins, list_max_ept, list_papr = [], [], [], []

    model.load(out)
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(exec_limit):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()

        # 記録
        list_action.append(action)
        list_theta_k_bins.append(env.unwrapped.theta_k_bins)
        list_max_ept.append(env.unwrapped.max_ept)
        list_papr.append(env.unwrapped.papr_db)

    _output(list_step, list_epi, list_action, list_theta_k_bins, list_max_ept, list_papr)

def _output(list_step, list_epi, list_action, \
            list_theta_k_bins, list_max_ept, list_papr) -> None:
    import pandas as pd
    data: pd.DataFrame = pd.DataFrame({
            'TimeStep': list_step,
            'Episode': list_epi,
            'Action': list_action,
            'theta_k_bin': list_theta_k_bins,
            'EP(t) [W]': list_max_ept,
            'PAPR [dB]': list_papr
        })
    data.to_csv(f'{out}.csv', index=False)

    min_i = list_papr.index(min(list_papr))
    text: str = f'theta_k: {list_theta_k_bins[min_i]}\nEP(t): {list_max_ept[min_i]} W / {list_papr[min_i]} dB\n'

    with open(f'{out}.txt', encoding="utf-8", mode='w') as file:
        file.write(text)
    print(text)
