from torch.utils.data import Dataset, DataLoader
from dataset.data_utils import *
import sys
import json
import copy
from tqdm import tqdm
import random

class E2EDataset(Dataset):
	
	def __init__(self, config, json_path, tokenizer, copy_vocab, is_training=False):
		super(E2EDataset, self).__init__()

		self.config = config
		self.tokenizer = tokenizer
		self.is_training = is_training
		self.copy_vocab = copy_vocab
		np.set_printoptions(threshold=sys.maxsize)

		self.keyword_norm = {'eatType': 'type', 'familyFriendly': 'family friendly', 'priceRange': 'price range'}
		self.key_index = {'type': 1, 'family friendly': 3, 'price range': 4, 'near': 5, 'name': 6, 'food': 7, 'area': 8, 'customer rating': 9}

		self.read_content(json_path)

	def __len__(self):
		return len(self.record)

	def __getitem__(self, index):
		ins_id, enc_input, enc_cls, mf, out, gt, gt_mr = self.record[index]

		data_item = {
			"ins_id": ins_id,
			"encoder_input_ids": enc_input,
			"encoder_class": enc_cls,
			"gt": gt,
			"gt_mr": gt_mr,
			"mention_flag": mf
		}
		if out is not None:
			data_item['cap'] = out

		return data_item

	def get_mention_index(self, cls_index, list_v):
		for (_, fg_index) in self.copy_vocab.d_to_w_group[cls_index]:
			fg_ch_list = self.copy_vocab.token_fg_w[fg_index]
			s1 = '&'.join([str(f) for f in fg_ch_list])
			for ch_idx, first_ch in enumerate(list_v):
				if first_ch == fg_ch_list[0]:
					s2 = '&'.join([str(f) for f in list_v[ch_idx: ch_idx + len(fg_ch_list)]])
					if s1 == s2:
						return ch_idx, ch_idx + len(fg_ch_list)
		return -1,-1

	def read_content(self, json_path):
		print("reading data from %s ..." % json_path)

		self.record = []

		with open(json_path) as out:
			instances = json.loads(out.read())

		total_input = 0
		match_input = 0
		type_info_ratio = {}
		for ins_id, instance in tqdm(enumerate(instances)):
			gt = copy.deepcopy(instance['ref'])
			gt_mr = []
			new_mr_input = []
			for mr in instance['mr']:
				each_mr = mr.replace('[', ' ')
				each_mr = each_mr.replace(']', ' ')
				each_mr = each_mr.strip()
				words = each_mr.split()
				if words[0] in self.keyword_norm:
					tag = self.keyword_norm[words[0]]
					value = ' '.join([x.strip() for x in words[1:]])
				else:
					tag = words[0] if not words[0] == 'customer' else ' '.join(words[:2])
					value = ' '.join([x.strip() for x in words[1:]]) if not words[0] == 'customer' else ' '.join([x.strip() for x in words[2:]])
				new_mr_input.append(tag + ' ' + value)
				gt_mr.append((mr, value))

			input_cls_info = []
			for (text, gt_m) in zip(new_mr_input, gt_mr):
				cls_id = self.copy_vocab.word_to_category_id[gt_m[0]]
				m_input = self.tokenizer(text, return_tensors="np")['input_ids'][0, :-1].tolist()
				input_cls_info.append((cls_id, m_input, gt_m[1]))

			encoder_input = []
			encoder_cls = []
			for (cls_id, m_input, _) in input_cls_info:
				encoder_input += m_input
				encoder_cls += [cls_id] * len(m_input)
			encoder_input.append(self.tokenizer.eos_token_id)
			encoder_cls.append(0)
			encoder_input = np.array(encoder_input, dtype=np.int64)
			encoder_cls = np.array(encoder_cls, dtype=np.int64)

			if not self.is_training:
				mention_flag = np.array(encoder_cls > 0, dtype=np.int64)
				mention_flag = mention_flag[np.newaxis, :]
				self.record.append((ins_id, encoder_input, encoder_cls, mention_flag, None, gt, gt_mr))
			else:
				for v in instance['ref']:
					ref = v
					v = ' '.join(v.split()).lower()
					v = self.tokenizer(v, return_tensors="np")['input_ids'][0, :self.config.max_generation_len]
					list_v = v.tolist()

					mentioned_cls_pos = []
					for (cls_id, m_input, name) in input_cls_info:
						s_pos, e_pos = self.get_mention_index(cls_id, list_v)
						mentioned_cls_pos.append((s_pos, e_pos, cls_id))
						total_input += 1
						if s_pos >= 0 and e_pos >= 0:
							match_input += 1
						# else:
						# 	print(self.copy_vocab.d_to_w_group[cls_id])
						# 	print(ref)
						# 	print("----------")

					encoder_input = np.array(encoder_input, dtype=np.int64)
					encoder_cls = np.array(encoder_cls, dtype=np.int64)
					mention_flag = np.zeros((v.shape[0], encoder_input.shape[0]), dtype=np.int64)

					for (s_pos, e_pos, cls_id) in mentioned_cls_pos:
						for e_index in range(encoder_cls.shape[0]):
							if encoder_cls[e_index] == cls_id:
								if e_pos >= 0:
									mention_flag[:e_pos, e_index] = 1
									if not self.config.static_mf:
										mention_flag[e_pos:, e_index] = 2
									else:
										mention_flag[e_pos:, e_index] = 1
								else:
									mention_flag[:, e_index] = 0 if not self.config.use_mf_merged else 1
					self.record.append((ins_id, encoder_input, encoder_cls, mention_flag, v, gt, gt_mr))

		if self.is_training:
			random.shuffle(self.record)
			print("Match Ratio %.2f" % (100 * match_input / total_input))


def data_wrapper(dataset):
	new_dataset = {'gt': [d['gt'] for d in dataset], 'gt_mr': [d['gt_mr'] for d in dataset], 'ins_id': [d['ins_id'] for d in dataset]}

	encoder_input_ids, encoder_mask = process_tensor([d['encoder_input_ids'] for d in dataset], 0, output_mask=True)
	encoder_class = process_tensor([d['encoder_class'] for d in dataset], 0, output_mask=False)
	new_dataset['encoder_input_ids'] = encoder_input_ids
	new_dataset['encoder_mask'] = encoder_mask
	new_dataset['encoder_cls'] = encoder_class

	max_gen_len = 1
	if 'cap' in dataset[0]:
		cap_decoder_input_ids, cap_decoder_mask = process_tensor([d['cap'] for d in dataset], 0, output_mask=True)
		cap_decoder_input_ids[cap_decoder_mask == 0] = -100
		new_dataset['cap_decoder_input_ids'] = cap_decoder_input_ids
		max_gen_len = cap_decoder_input_ids.size(1)

	batch_size = len(dataset)
	max_encoder_len = encoder_input_ids.size(1)
	mention_flag = np.zeros((batch_size, max_gen_len, max_encoder_len), dtype=np.int64)
	for i, d in enumerate(dataset):
		mention_flag[i, :d['mention_flag'].shape[0], :d['mention_flag'].shape[1]] = d['mention_flag']
	new_dataset['mention_flag'] = torch.from_numpy(mention_flag)

	return new_dataset


def get_data_loader(dataset, batch_size):
    collate_fn = lambda d: data_wrapper(d)
    return DataLoader(dataset, 
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )



