from typing import Any, List
from yacs.config import CfgNode as CN

class Config(object):

	def __init__(self, config_yaml: str, config_override: List[Any] = []):
		self._C = CN()
		self._C.random_seed = 0
		self._C.train_path = ""
		self._C.dev_path = ""
		self._C.test_path = ""
		self._C.train_obj_h5_path = ""
		self._C.dev_obj_h5_path = ""
		self._C.test_obj_h5_path = ""
		self._C.train_copy_obj_h5_path = ""
		self._C.dev_copy_obj_h5_path = ""
		self._C.test_copy_obj_h5_path = ""
		self._C.object_blacklist_path = ""
		self._C.copy_vocab_path = ""
		self._C.lm_type = "t5-large"
		self._C.vocab_size = 0
		self._C.use_pointer = False
		self._C.batch_size = 192
		self._C.max_epoch = 20
		self._C.gradient_accumulation_steps = 1
		self._C.checkpoint_every_step = 1000
		self._C.weight_decay = 0.0
		self._C.adam_epsilon = 1e-8
		self._C.learning_rate = 5e-5
		self._C.warmup_step = 400
		self._C.num_training_steps = 0
		self._C.grad_clip_value = 0
		self._C.use_mention_flag = False
		self._C.mention_flag_state = 3
		self._C.max_generation_len = 25
		self._C.relation_map_path = ""
		self._C.entity_map_path = ""
		self._C.word_norm_jsonpath = ""
		self._C.enable_visual = False
		self._C.roi_dim = 2048
		self._C.box_dim = 8
		self._C.use_copy_obj = False
		self._C.rm_dumplicated_caption = False
		self._C.shuffle_data = False
		self._C.rm_punctuation = False
		self._C.external_eval = False
		self._C.relative_pos_num = 0
		self._C.use_orginal_enc_pos_embs = False
		self._C.freeze_param = True
		self._C.freeze_enc_pos_param = True
		self._C.decode_constrain = ""
		self._C.static_mf = False
		self._C.do_pretrain_lm_init = True
		self._C.use_mf_scalar = False
		self._C.use_mf_merged = False


		# Override parameter values from YAML file first, then from override list.
		self._C.merge_from_file(config_yaml)
		self._C.merge_from_list(config_override)

		# Make an instantiated object of this class immutable.
		self._C.freeze()

	def dump(self, file_path: str):
		self._C.dump(stream=open(file_path, "w"))

	def __getattr__(self, attr: str):
		return self._C.__getattr__(attr)

	def __str__(self):
		return _config_str(self)

	def __repr__(self):
		return self._C.__repr__()


def _config_str(config: Config) -> str:
    r"""
    Collect a subset of config in sensible order (not alphabetical) according to phase. Used by
    :func:`Config.__str__()`.

    Parameters
    ----------
    config: Config
        A :class:`Config` object which is to be printed.
    """
    _C = config

    __C: CN = CN({"RANDOM_SEED": _C.random_seed})
    common_string: str = str(__C) + "\n"

    return common_string