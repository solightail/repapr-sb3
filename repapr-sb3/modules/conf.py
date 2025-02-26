import tomllib

class Conf(object):
    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Conf, cls).__new__(cls)
        return cls._instance

    def __init__(self, filepath=None, *args, **karg):
        if not hasattr(self, "_init"):
            self._init = True

            # 設定ファイルの読み込み
            with open(filepath, 'rb') as file:
                cfg = tomllib.load(file)

            self.algorithm: str = cfg['algorithm']
            self.mode: int = cfg['mode']

            # 入力変数
            self.tones: int = cfg['input']['tones']
            self.del_freq: float = cfg['input']['del_freq']
            self.del_time: float = cfg['input']['del_time']
            self.amp: float = cfg['input']['amp']
            self.theta_k_model: str = cfg['input']['theta_k_model']
            self.phase_value: float = cfg['input']['phase_value']
            self.manual: list = cfg['input']['manual']

            # 追加処理
            self.rt_ept_graph: bool = cfg['addproc']['rt_ept_graph']
            self.rt_phase_cricle: bool = cfg['addproc']['rt_phase_cricle']
            self.notify: bool = cfg['addproc']['notify']

            # 環境パラメータ
            self.continuous: bool = cfg['env']['param']['continuous']
            self.const_first_phase: bool = cfg['env']['param']['const_first_phase']
            self.action_control: int = cfg['env']['param']['action_control']
            self.total_timesteps: int = cfg['env']['param']['total_timesteps']
            self.max_episode_steps: int = cfg['env']['param']['max_episode_steps']

            # 環境パラメータ（継承）
            self.reduction_ratio: float = cfg['env']['param']['inherit']['reduction_ratio']
            self.stack_limit: int = cfg['env']['param']['inherit']['stack_limit']
            self.init_eval: int = cfg['env']['param']['inherit']['init_eval']
            self.change_eval: int = cfg['env']['param']['inherit']['change_eval']
            self.n_inherit: int = cfg['env']['param']['inherit']['n_inherit']

            # 観測・報酬パラメータ
            self.observation_items: dict = cfg['env']['observation']
            self.eval_metrics: str = cfg['env']['reward']['eval_metrics']
            self.eval_model: str = cfg['env']['reward']['eval_model']

            # ハイパーパラメータ
            self.N: int = cfg['hyper']['N']
            self.batch_size: int = cfg['hyper']['batch_size']
            self.n_epochs: int = cfg['hyper']['n_epochs']
            self.alpha: float = cfg['hyper']['alpha']
            self.gamma: float = cfg['hyper']['gamma']

            # 出力
            self.filepath: str = cfg['output']['filepath']
            # LINE設定
            self.line: dict = cfg['output']['line']
