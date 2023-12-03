from gymnasium.envs.registration import register
from gymnasium.wrappers.autoreset import AutoResetWrapper
from .modules.conf import Conf

conf_filepath = "repapr-ppo-cas/config.toml"
cfg = Conf(conf_filepath)

register(
    id='repapr-v0',
    entry_point='repapr-ppo-cas.modules.env:RePAPREnv',
    max_episode_steps=cfg.max_episode_steps,
    additional_wrappers=(AutoResetWrapper.wrapper_spec(),)
)