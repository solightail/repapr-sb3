import os
import gymnasium as gym

from .modules.conf import Conf

cfg = Conf()
out = f"{cfg.filepath}/{cfg.algorithm}/N{cfg.tones}/{cfg.eval_model}_{cfg.action_control}_LT{cfg.total_timesteps}_LE{cfg.max_episode_steps}"
# unuse names
# _G{str(cfg.gamma).replace('.', '')}
env = gym.make("repapr-v0", render_mode="debug")

# 学習アルゴリズム選択
match cfg.algorithm:
    case "PPO":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1, gamma=cfg.gamma, n_steps=cfg.N, batch_size=cfg.batch_size)
    case "SAC":
        from stable_baselines3 import SAC
        model = SAC("MlpPolicy", env, verbose=1, gamma=cfg.gamma, batch_size=cfg.batch_size)
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
        case 3:
            inherit_exec()
        case 4:
            inherit_learn_exec()
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
    message = f"repapr-sb3 / {end.time().isoformat(timespec='seconds')}\n{cfg.algorithm} N{cfg.tones}_LT{cfg.total_timesteps}_LE{cfg.max_episode_steps}\n{in_msg}"
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
        if cfg.algorithm == 'SAC':
            action, _states = model.predict(obs, deterministic=True)
        else:
            action, _states = model.predict(obs)
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
    text: str = f'\ntheta_k: {list_theta_k_bins[min_i]}\nEP(t): {list_max_ept[min_i]} W / {list_papr[min_i]} dB'

    with open(f'{out}.txt', encoding="utf-8", mode='w') as file:
        file.write(text)
    print(text)

def inherit_exec() -> None:
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
    for _ in range(exec_loop):
        for _ in range(cfg.max_episode_steps):
            if cfg.algorithm == 'SAC':
                action, _states = model.predict(obs, deterministic=True)
            else:
                action, _states = model.predict(obs)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render()

            # 記録
            list_action.append(action)
            list_theta_k_bins.append(env.unwrapped.theta_k_bins)
            list_max_ept.append(env.unwrapped.max_ept)
            list_papr.append(env.unwrapped.papr_db)

        cfg.theta_k_model = 'manual'
        min_i = list_papr.index(min(list_papr))
        env.unwrapped.set_options(dict(manual_theta_k=list_theta_k_bins[min_i]))
        obs = vec_env.reset()

    _output(list_step, list_epi, list_action, list_theta_k_bins, list_max_ept, list_papr)

def inherit_learn_exec() -> None:
    from datetime import datetime
    start = datetime.now()
    # 学習モデルの上書き確認
    if os.path.exists(f"{out}/0.zip"):
        if not input('Overwrite learned files? [Y/n] ') in ["Y", "y", "YES", "Yes", "yes"]:
            raise FileExistsError("Learned file already exists.")

    if cfg.notify is True: _notify(f"演算を開始します")

    # 変数初期化
    best_papr = None
    exec_i = 0
    step = 0
    papr_stack = 0
    papr_stacked = False
    papr_renew = 0
    exec_loop = 50
    list_step, list_epi, list_action, list_theta_k_bins, list_max_ept, list_papr = [], [], [], [], [], []

    while papr_renew in range(exec_loop):
        # 学習
        match cfg.algorithm:
            case "PPO":
                model.learn(total_timesteps=cfg.max_episode_steps*8, progress_bar=True)
            case "SAC":
                model.learn(total_timesteps=cfg.max_episode_steps*8, log_interval=4, progress_bar=True)
            case _:
                raise ValueError("A non-existent algorithm is selected.")
        model.save(f"{out}/{exec_i}")

        # 学習データ読み込み
        model.load(f"{out}/{exec_i}")
        vec_env = model.get_env()
        obs = vec_env.reset()
        for _ in range(cfg.max_episode_steps):
            step += 1
            if cfg.algorithm == 'SAC':
                action, _states = model.predict(obs, deterministic=True)
            else:
                action, _states = model.predict(obs)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render()

            # 記録
            list_step.append(step)
            list_epi.append(papr_renew)
            list_action.append(action)
            list_theta_k_bins.append(env.unwrapped.theta_k_bins)
            list_max_ept.append(env.unwrapped.max_ept)
            list_papr.append(env.unwrapped.papr_db)

        cfg.theta_k_model = 'manual'
        if best_papr is None: best_papr = min(list_papr)
        if best_papr == min(list_papr):
            papr_stack += 1
            if papr_stack == 10:
                papr_stack = 0
                papr_stacked = True
        if best_papr > min(list_papr) or papr_stacked is True:
            papr_renew += 1
            papr_stack = 0
            papr_stacked = False
            best_papr = min(list_papr)
            print(f"renewal: {papr_renew} / PAPR: {best_papr} / metrics: {env.unwrapped.eval_metrics}")

            if papr_renew >= 5:
                next_max_phase = env.unwrapped.max_phase * cfg.reduction_ratio
                min_i = list_papr.index(best_papr)
                next_theta_k = list_theta_k_bins[min_i]
                env.unwrapped.set_options(dict(max_phase=next_max_phase, manual_theta_k=next_theta_k))
            if papr_renew == 15:
                env.unwrapped.eval_metrics = "db"
                env.unwrapped.eval_model = "Square"

        obs = vec_env.reset()
        exec_i += 1

    _output(list_step, list_epi, list_action, list_theta_k_bins, list_max_ept, list_papr)
    end = datetime.now()
    print(f'PAPR[dB]: {best_papr}  経過時間:{end-start}')
    if cfg.notify is True: _notify(f"学習が完了しました")
