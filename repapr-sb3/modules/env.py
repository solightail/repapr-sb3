from .conf import Conf

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
import gymnasium as gym
from gymnasium import spaces

class RePAPREnv(gym.Env):
    """
    gymnasium v0.29.1 準拠
    reference -> Pendulum-v1
    """
    metadata = {
        "render_modes": ["human", "debug"],
        'video.frames_per_second' : 30,
        'render_fps': 30
    }

    def __init__(self, render_mode: Optional[str] = None):
        # Load config
        cfg = Conf()
        self.tones: int = cfg.tones
        self.del_freq: float = cfg.del_freq
        self.del_time: float = cfg.del_time
        self.amp: float = cfg.amp
        self.theta_k_model: str = cfg.theta_k_model
        self.phase_value: float = cfg.phase_value
        self.manual: list = cfg.manual
        self.observation_items: dict = cfg.observation_items
        self.eval_metrics: str = cfg.eval_metrics
        self.eval_model: str = cfg.eval_model
        self.continuous: bool = cfg.continuous
        self.const_first_phase: bool = cfg.const_first_phase
        self.action_control: int = cfg.action_control
        self.rt_graph: bool = cfg.rt_graph

        self.render_mode = render_mode

        # 変数初期化
        self.amse = None
        self._options = None
        self.time_arr: np.ndarray = np.arange(0.0, 1.0 + self.del_time, self.del_time)
        if self.render_mode is not None and self.rt_graph is True: self.init_rt_graph: bool = True

        # action スペース設定
        # 1 => 2π / -1 ~ 1 であれば 0.5 で少なくとも 2π 分は変位可能
        if self.action_control == 0:
            self.max_phase = 0.5 / 20
            self.action_low = -self.max_phase
            self.action_high = self.max_phase
        else:
            self.max_phase = 1.0
            self.action_low = 0.0
            self.action_high = self.max_phase
        if self.const_first_phase is True:
            _action_shape = self.tones-1
        else:
            _action_shape = self.tones

        if self.continuous is True:
            self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape=(_action_shape,), dtype=np.float64)
        else:
            self.action_space = spaces.Discrete(3*self.tones)

        # observation の次元設定とスペース設定の準備
        observation_low = []
        observation_high = []
        self.input_dims = 0
        minp0t = (1/(2*self.amp**2))*((self.tones)-(self.tones*self.amp**2))
        maxp0t = (1/(2*self.amp**2))*((self.tones**2)-(self.tones*self.amp**2))
        if self.observation_items['theta_k'] is True:
            self.input_dims += self.tones
            for _ in range(self.tones):
                observation_low.append(0.0)
                observation_high.append(1.0)
        if self.observation_items['theta_k_diff']:
            self.input_dims += self.tones-1
            for _ in range(self.tones-1):
                observation_low.append(0.0)
                observation_high.append(1.0)
        if self.observation_items['ept'] is True:
            self.input_dims += len(self.time_arr)
            for _ in range(len(self.time_arr)):
                observation_low.append(0.0)
                observation_high.append(self.tones**2)
        if self.observation_items['papr'] is True:
            self.input_dims += 1
            observation_low.append(10*np.log10(1+((2*minp0t)/self.tones)))
            observation_high.append(10*np.log10(1+((2*maxp0t)/self.tones)))
        if self.observation_items['mse'] is True:
            self.input_dims += 1
            observation_low.append(0.0)
            observation_high.append(maxp0t**2)
        if self.observation_items['peaks_height'] is True:
            if self.eval_metrics == 'abs':
                match self.eval_model:
                    case 'USo_v1': self.obs_n_ph = 1
                    case 'USt_v1': self.obs_n_ph = 2
                    case 'UFtSt_v1': self.obs_n_ph = 2
                    case 'BSo_v1': self.obs_n_ph = 2
                    case 'BSt_v1': self.obs_n_ph = 4
                    case 'BFt_v1': self.obs_n_ph = 4
            else: self.obs_n_ph = 4
            self.input_dims += self.obs_n_ph
            for _ in range(self.obs_n_ph):
                observation_low.append(0.0)
                observation_high.append(maxp0t)
        if self.observation_items['len_both_peaks'] is True:
            self.input_dims += 1
            observation_low.append(0.0)
            observation_high.append(np.inf) #2*(self.tones-1)

        self.input_dims = (self.input_dims, )

        observation_low = np.array(observation_low, dtype=np.float64)
        observation_high = np.array(observation_high, dtype=np.float64)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float64)


    def step(self, action):
        action = np.clip(action, self.action_low, self.action_high)
        if self.action_control == 0:
            if self.const_first_phase is True:
                if self.theta_k_model == "manual":
                    self.theta_k_bins = np.insert(np.mod(np.add(self.theta_k_bins[1:], action), 1.0), 0, self.manual[0])
                else:
                    self.theta_k_bins = np.insert(np.mod(np.add(self.theta_k_bins[1:], action), 1.0), 0, self.phase_value)
            else:
                self.theta_k_bins = np.mod(np.add(self.theta_k_bins, action), 1.0)
        else:  #self.action_control == 1:
            if self.const_first_phase is True:
                if self.theta_k_model == "manual":
                    self.theta_k_bins = np.insert(np.mod(action, 1.0), 0, self.manual[0])
                else:
                    self.theta_k_bins = np.insert(np.mod(action, 1.0), 0, self.phase_value)
            else:
                self.theta_k_bins = np.mod(action, 1.0)

        self.theta_k_bins_diffs = [np.mod(self.theta_k_bins[i+1]-self.theta_k_bins[i], 1) for i in range(self.tones-1)]

        self.ept_arr, self.max_ept, self.papr_w, self.papr_db = self._get_ept_arr()

        self.known_amse, self.known_peaks = False, False    # 再計算防止
        new_obs, new_rew = self._get_observation(), self._get_reward()

        if self.render_mode == "human":
            self.render(action)
        elif self.render_mode == "debug":
            self.render(action)

        """
        terminated と truncated は False 固定
        エピソードに対するステップ上限は、環境登録にて max_episode_steps で設定済み
        """

        return new_obs, new_rew, False, False, {}

    def _get_observation(self):
        observation = []
        if self.observation_items['theta_k'] is True:
            observation += self.theta_k_bins.tolist()
        if self.observation_items['theta_k_diff'] is True:
            observation += self.theta_k_bins_diffs
        if self.observation_items['ept'] is True:
            # 僅かに負の値となることがあるためクリップ
            observation += np.clip(self.ept_arr, 0.0, self.tones**2).tolist()
        if self.observation_items['papr'] is True:
            observation.append(self.papr_db)
        if self.observation_items['mse'] is True:
            if self.known_amse is False: self.amse = self._calc_mse(self.ept_arr)
            observation.append(self.amse)
        if self.observation_items['peaks_height'] is True:
            if self.known_peaks is False: self._calc_peaks()
            if self.obs_n_ph == 1:
                observation += [self.up1h]
            elif self.obs_n_ph == 2:
                observation += [self.up1h, self.lo1h]
            else:
                observation += [self.up1h, self.up2h, self.lo1h, self.lo2h]
        if self.observation_items['len_both_peaks'] is True:
            if self.known_peaks is False: self._calc_peaks()
            observation.append(len(self.upper_peaks_heights)+len(self.lower_peaks_heights))
        return np.array(observation, dtype=np.float64)

    def _get_reward(self):
        match self.eval_metrics:
            case 'db':
                match self.eval_model:
                    case 'Raw':
                        reward = -self.papr_db
                    case 'Double':
                        reward = -self.papr_db*2
                    case 'Square':
                        reward = -self.papr_db**2
            case 'abs':
                match self.eval_model:
                    case 'USo_v1':
                        if self.known_peaks is False: self._calc_peaks()
                        self.obs_peaks_height = [self.up1h]
                        reward = (self.tones+2 - self.up1h)
                    case 'USt_v1':
                        if self.known_peaks is False: self._calc_peaks()
                        self.obs_peaks_height = [self.up1h, self.up2h]
                        reward = (self.tones+2 - self.up1h) + (self.tones+2 - self.up2h)
                    case 'UFtSt_v1':
                        if self.known_peaks is False: self._calc_peaks()
                        self.obs_peaks_height = [self.up1h, self.up2h]
                        reward = (self.tones+2 - self.up1h) + (self.tones+2 - self.up2h) - (self.up1h - self.up2h)
                    case 'BSo_v1':
                        if self.known_peaks is False: self._calc_peaks()
                        self.obs_peaks_height = [self.up1h, self.lo1h]
                        reward = (self.tones+2 - self.up1h) + (self.lo1h - self.tones-2)
                    case 'BSt_v1':
                        if self.known_peaks is False: self._calc_peaks()
                        self.obs_peaks_height = [self.up1h, self.up2h, self.lo1h, self.lo2h]
                        reward = (self.tones+2 - self.up1h) + (self.lo1h - self.tones-2) + (self.tones+2 - self.up2h) + (self.lo2h - self.tones-2)
                    case 'BFt_v1':
                        if self.known_peaks is False: self._calc_peaks()
                        self.obs_peaks_height = [self.up1h, self.up2h, self.lo1h, self.lo2h]
                        reward = (self.tones+2 - self.up1h) + (self.lo1h - self.tones-2) + (self.tones+2 - self.up2h) + (self.lo2h - self.tones-2) - (self.up1h - self.up2h) - (self.lo2h - self.lo1h)
            case 'mse':
                match self.eval_model:
                    case 'AMSE_v0':
                        if self.known_amse is False: self.amse = self._calc_mse(self.ept_arr)
                        reward = -self.amse
                    case 'BMSE_v0':
                        if self.known_peaks is False: self._calc_peaks()
                        both_peaks_heights = np.append(self.upper_peaks_heights, self.lower_peaks_heights)
                        reward = -self._calc_mse(both_peaks_heights)
        return reward

    def _calc_peaks(self) -> None:
        self.known_peaks = True
        # ピーク値取得
        upper_peaks, _ = find_peaks(self.ept_arr, distance=10, plateau_size=1)
        lower_peaks, _ = find_peaks(-self.ept_arr, distance=10, plateau_size=1)

        # find_peaks は t = 0 にピークがある場合を考慮していない。
        # 1周期でこれを判断することは不可能であるため、2周期分で判断をし、1周期分に戻す作業を行う。
        two_cycle_ept_arr = np.concatenate([self.ept_arr, self.ept_arr[1:]])
        two_cycle_upper_peaks, _ = find_peaks(two_cycle_ept_arr, distance=10, plateau_size=1)
        two_cycle_lower_peaks, _ = find_peaks(-two_cycle_ept_arr, distance=10, plateau_size=1)
        period: int = len(self.ept_arr)-1
        if (period in two_cycle_upper_peaks):
            upper_peaks = np.concatenate([[0], upper_peaks, [period]])
        if (period in two_cycle_lower_peaks):
            lower_peaks = np.concatenate([[0], lower_peaks, [period]])

        # find_peaks 後処理
        # 最大ピークおよび準最大ピークの検出
        self.upper_peaks_heights = np.take(self.ept_arr, upper_peaks)
        max2_reverse_upper_peaks = upper_peaks[np.argsort(self.upper_peaks_heights)][-2:]
        self.lower_peaks_heights = np.take(self.ept_arr, lower_peaks)
        max2_reverse_lower_peaks = lower_peaks[np.argsort(self.lower_peaks_heights)][:2]

        # upper_peaks 平均・最大ピーク・準最大ピーク
        self.upah = np.average(np.take(self.ept_arr, upper_peaks))
        self.up1t = np.take(self.time_arr, max2_reverse_upper_peaks[1])
        self.up1h = np.take(self.ept_arr, max2_reverse_upper_peaks[1])
        self.up2t = np.take(self.time_arr, max2_reverse_upper_peaks[0])
        self.up2h = np.take(self.ept_arr, max2_reverse_upper_peaks[0])

        # lower_peaks 平均・最大ピーク・準最大ピーク
        self.loah = np.average(np.take(self.ept_arr, lower_peaks))
        self.lo1t = np.take(self.time_arr, max2_reverse_lower_peaks[0])
        self.lo1h = np.take(self.ept_arr, max2_reverse_lower_peaks[0])
        self.lo2t = np.take(self.time_arr, max2_reverse_lower_peaks[1])
        self.lo2h = np.take(self.ept_arr, max2_reverse_lower_peaks[1])

    def _calc_mse(self, input_arr):
        criterion = nn.MSELoss()
        input_tensor = torch.tensor(input_arr, requires_grad=True).double()
        target_array = np.full(len(input_arr), self.tones)
        target_tensor = torch.tensor(target_array).double()
        loss = criterion(input_tensor, target_tensor)
        loss.backward()
        mse = loss.detach().numpy().copy()
        return mse


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self._options is None:
            self.theta_k_bins = self._get_theta_k()
        else:
            # マニュアル値はオプションより入力 (set_optionsメソッドよりdict型で入力)
            self.theta_k_bins = self._options.get("manual_theta_k") if "manual_theta_k" in self._options else self._get_theta_k()

        self.theta_k_bins_diffs = [np.mod(self.theta_k_bins[i+1]-self.theta_k_bins[i], 1) for i in range(self.tones-1)]

        # observation 計算
        self.ept_arr, self.max_ept, self.papr_w, self.papr_db = self._get_ept_arr()
        self.known_peaks = False

        self.known_amse = False  # 再計算防止
        new_obs = self._get_observation()

        if self.render_mode == "human":
            self.render()
        elif self.render_mode == "debug":
            self.render()

        return new_obs, {}

    def set_options(self, options):
        # Set options for the next reset
        self._options = options

    def _get_theta_k(self):
        from .theta_k_model import Narahashi, Newman, Unify, Random, Manual, AContext
        # アルゴリズム選択
        match self.theta_k_model:
            case 'narahashi':
                strategy = Narahashi()
            case 'newman':
                strategy = Newman()
            case 'unify':
                strategy = Unify()
            case 'random':
                strategy = Random()
            case 'manual':
                strategy = Manual(self.manual)

        # theta_k 計算
        algo_context = AContext(strategy)
        return algo_context.calc_algo()

    def _get_ept_arr(self):
        from .calc import Formula, FEPtA
        formula = Formula(self.tones, self.del_freq, self.amp)
        fepta = FEPtA(formula, self.del_time)
        return fepta.get_ept_papr(self.theta_k_bins)


    def render(self, action=None):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="human")'
            )
            return

        if self.render_mode == "human":
            if self.rt_graph is True:
                if self.init_rt_graph is True:
                    from .utils import rt_plot_init, rt_circle_init
                    self.lines, self.plot_text_bl, self.plot_text_br = rt_plot_init(self.time_arr, self.ept_arr, self.papr_db, self.amse)
                    self.circle_lines = rt_circle_init(self.theta_k_bins_diffs)
                    self.init_rt_graph = False

                from .utils import rt_plot_reload_line, rt_plot_reload_text_br, rt_circle_reload_line, rt_pause_plot
                # rt_graph リセット
                rt_plot_reload_line(self.lines, self.time_arr, self.ept_arr)
                rt_plot_reload_text_br(self.plot_text_br, self.papr_db, self.amse)
                rt_circle_reload_line(self.circle_lines, self.theta_k_bins_diffs)
                rt_pause_plot()

        else:  # mode == "debug":
            if self.rt_graph is True:
                if self.init_rt_graph is True:
                    from .utils import rt_plot_init, rt_circle_init
                    self.lines, self.plot_text_bl, self.plot_text_br = rt_plot_init(self.time_arr, self.ept_arr, self.papr_db, self.amse)
                    self.circle_lines = rt_circle_init(self.theta_k_bins_diffs)
                    self.init_rt_graph = False

                from .utils import rt_plot_reload_line, rt_plot_reload_text_br, rt_circle_reload_line, rt_pause_plot
                # rt_graph リセット
                rt_plot_reload_line(self.lines, self.time_arr, self.ept_arr)
                rt_plot_reload_text_br(self.plot_text_br, self.papr_db, self.amse)
                rt_circle_reload_line(self.circle_lines, self.theta_k_bins_diffs)
                if action is not None:
                    print("-----------------------------------------------")
                    print(f"action : {action}")
                    print(f"theta_k: {self.theta_k_bins}")
                else:
                    print("-------------------- reset --------------------")
                    print(f"theta_k: {self.theta_k_bins}")
                rt_pause_plot()

    def close():
        from .utils import close_plot
        close_plot()
