from torch.utils.data import Dataset, DataLoader
from dataset.reader import CocoCaptionsReader, ImageFeaturesReader
from tqdm import tqdm
import json
import random
import sys
from dataset.data_utils import *
import numpy as np

class COCODataset(Dataset):

	def __init__(self, config, h5_path, tokenizer, copy_vocab, attachable_index, caption_path=None, copy_h5_path=None, is_training=False, in_memory=False, cbs_class_path=None):
		if caption_path is not None:
			self._captions_reader = CocoCaptionsReader(caption_path, config.word_norm_jsonpath if len(config.word_norm_jsonpath) > 0 else None, rm_dumplicated_caption=config.rm_dumplicated_caption, shuffle=config.shuffle_data, is_train=is_training, rm_punctuation=config.rm_punctuation)
		else:
			self._captions_reader = None

		np.set_printoptions(threshold=sys.maxsize)

		self._image_features_reader = ImageFeaturesReader(h5_path)
		if config.use_copy_obj:
			self._copy_image_features_reader = ImageFeaturesReader(copy_h5_path, start_index=1601)
		self.config = config
		self.is_training = is_training
		self.copy_vocab = copy_vocab
		self.tokenizer = tokenizer
		self.attachable_index = attachable_index
		self.cbs_class = None
		if cbs_class_path is not None:
			self.cbs_class = {}
			with open(cbs_class_path) as out:
				for line in out:
					line = line.strip()
					items = line.split(',')
					self.cbs_class[int(items[0])] = sorted([int(v) for v in items[1:]])

		self._image_ids = sorted(list(self._image_features_reader._map.keys()))
		self.obj_cache = {}
		self.cap_cache = {}
		self.global_obj_cache = {}

		if len(config.object_blacklist_path) > 0:
			with open(config.object_blacklist_path) as out:
				blacklist = json.load(out)
				full_list = blacklist['blacklist_categories'] + (blacklist['val_blacklist_categories'] if not is_training else [])
			self._blacklist_categories = set([s.lower() for s in full_list])
		else:
			self._blacklist_categories = None

		self.img_index = self.tokenizer("<unk>", return_tensors="np")['input_ids'][0, 0]
		self.background_index = self.tokenizer("background", return_tensors="np")['input_ids'][0, 0]

		if in_memory or (not is_training):
			self._image_features_reader.open_h5_file()
			if config.use_copy_obj:
				self._copy_image_features_reader.open_h5_file()

			for index in tqdm(range(len(self._captions_reader))):
				img_id, cap, _ = self._captions_reader[index]
				if img_id not in self.obj_cache:
					self.process_obj(img_id)
				if self._captions_reader is not None:
					if img_id not in self.cap_cache or cap not in self.cap_cache[img_id]:
						self.process_cap(img_id, cap)
			for img_id in tqdm(self.cap_cache):
				self.process_global_cap(img_id)

			self._image_features_reader.close_h5_file()
			if config.use_copy_obj:
				self._copy_image_features_reader.close_h5_file()

	def __len__(self):
		if self.is_training:
			return len(self._captions_reader)
		else:
			return len(self._image_ids)

	def process_global_cap(self, img_id):
		if self._captions_reader is not None:
			obj_count = {}
			for _, cap in self.cap_cache[img_id].items():
				for c in cap['used_cls']:
					if c not in obj_count:
						obj_count[c] = 0
					obj_count[c] += 1
			sort_obj = sorted(obj_count.items(), key=lambda x: x[1], reverse=True)
			top_obj = [x[0] for x in sort_obj[:2]]
		else:
			top_obj = []

		mention_flag = np.zeros((1, self.obj_cache[img_id]['encoder_input_ids'].shape[0]), dtype=np.int64)
		if self.cbs_class is not None:
			top_obj = self.cbs_class[img_id]

		for index, ecls in enumerate(self.obj_cache[img_id]['encoder_cls'].tolist()):
			if ecls in top_obj:
				mention_flag[0, index] = 1
			elif ecls < 1601:
				mention_flag[0, index] = 3

		self.global_obj_cache[img_id] = mention_flag

	def process_cap(self, img_id, cap):
		if img_id not in self.cap_cache:
			self.cap_cache[img_id] = {}

		if cap not in self.cap_cache[img_id]:
			self.cap_cache[img_id][cap] = {}
			self.cap_cache[img_id][cap]['input_ids'] = self.tokenizer(cap.lower(), return_tensors="np")['input_ids'][0, :self.config.max_generation_len]

			mention_flag = np.zeros((self.cap_cache[img_id][cap]['input_ids'].shape[0], self.obj_cache[img_id]['encoder_input_ids'].shape[0]), dtype=np.int64)
			c_input_ids = self.cap_cache[img_id][cap]['input_ids'].tolist()
			en_cls = self.obj_cache[img_id]['encoder_cls'].tolist()
			visit_en_cls = []
			start_pos = {}
			for i, c in enumerate(en_cls):
				if c not in visit_en_cls:
					start_pos[len(visit_en_cls)] = i
					visit_en_cls.append(c)

			start_pos[len(visit_en_cls)] = len(en_cls)

			used_cls = []
			for j, cls_index in enumerate(visit_en_cls):
				if cls_index >= 1601: 
					found_word = False
					all_fgs = [fg_index for (_, fg_index) in self.copy_vocab.d_to_w_group[cls_index]]
					for fg_index in all_fgs:
						fg_ch_list = self.copy_vocab.token_fg_w[fg_index]
						s1 = '&'.join([str(f) for f in fg_ch_list])

						for ch_idx, first_ch in enumerate(c_input_ids):
							if first_ch == fg_ch_list[0]:
								s2 = '&'.join([str(f) for f in c_input_ids[ch_idx: ch_idx + len(fg_ch_list)]])
								if s1 == s2:
									if ch_idx + len(fg_ch_list) >= len(c_input_ids) - 1 or c_input_ids[ch_idx + len(fg_ch_list)] not in self.attachable_index:
										mention_flag[:ch_idx + len(fg_ch_list), start_pos[j]:start_pos[j+1]] = 1
										if not self.config.static_mf:
											mention_flag[ch_idx + len(fg_ch_list):, start_pos[j]:start_pos[j+1]] = 2
										else:
											mention_flag[ch_idx + len(fg_ch_list):, start_pos[j]:start_pos[j+1]] = 1
										used_cls.append(cls_index)
										found_word = True
										break

						if found_word: break
				else:
					mention_flag[:, start_pos[j]:start_pos[j+1]] = 3

			self.cap_cache[img_id][cap]['mention_flag'] = mention_flag
			self.cap_cache[img_id][cap]['used_cls'] = list(set(used_cls))

	def process_obj(self, img_id):
		image_features, box_np, class_np = self._image_features_reader[img_id]
		if self.config.use_copy_obj:
			copy_image_features, copy_box_np, copy_class_np = self._copy_image_features_reader[img_id]
			self.process_input_for_encoder(img_id, image_features, box_np, class_np, copy_image_features, copy_box_np, copy_class_np)
		else:
			self.process_input_for_encoder(img_id, image_features, box_np, class_np)

	def process_input_for_encoder(self, img_id, obj_features, obj_boxes, obj_cls, copy_obj_features=None, copy_obj_boxes=None, copy_obj_cls=None):
		obj_size = obj_features.shape[0]

		cls2objindex = {}
		for obj_index, cls_ in enumerate(obj_cls):
			if self._blacklist_categories is not None and self.copy_vocab.id_to_category[cls_].lower() in self._blacklist_categories:
				cls_ = 0
			if cls_ not in cls2objindex:
				cls2objindex[cls_] = []
			cls2objindex[cls_].append((obj_index, obj_boxes[obj_index, 6]))

		if self.config.use_copy_obj:
			for obj_index, cls_ in enumerate(copy_obj_cls):
				if self._blacklist_categories is not None and self.copy_vocab.id_to_category[cls_].lower() in self._blacklist_categories:
					cls_ = 0
				if cls_ not in cls2objindex:
					cls2objindex[cls_] = []
				cls2objindex[cls_].append((obj_index + obj_size, copy_obj_boxes[obj_index, 6]))
		
		for cls_ in cls2objindex:
			cls2objindex[cls_] = sorted(cls2objindex[cls_], key=lambda x: x[1])

		encoder_input_ids = []
		encoder_img_mask = []
		encoder_cls = []
		rel_position = []
		img_order = []

		key_order = sorted([k for k in cls2objindex.keys()])
		for cls_ in key_order:
			rel_position_list = []
			
			if cls_ == 0:
				input_ids = [self.background_index]
			else:
				input_ids = self.copy_vocab.token_class[cls_]
			
			for img_i in range(len(cls2objindex[cls_])):
				each_img_rel = [48] * len(cls2objindex[cls_])
				each_img_rel[img_i] = 0
				each_img_rel += [31 + get_position_emb_index(w_i + 1) for w_i in range(len(input_ids))]
				rel_position_list.append(each_img_rel)
			for word_i in range(len(input_ids)):
				each_word_rel = [49] * len(cls2objindex[cls_])
				each_word_rel += [0 if ii == word_i else get_position_emb_index(abs(ii - word_i), right=ii > word_i) for ii in range(len(input_ids))]
				rel_position_list.append(each_word_rel)
			rel_position_np = np.array(rel_position_list, dtype=np.int64)
			assert rel_position_np.shape[0] == rel_position_np.shape[1]
			rel_position.append(rel_position_np)

			sub_span = [self.img_index] * len(cls2objindex[cls_]) + input_ids
			encoder_input_ids += sub_span
			encoder_img_mask += [1] * len(cls2objindex[cls_]) + [0] * len(input_ids)
			encoder_cls += [cls_] * len(sub_span)
			img_order += [o[0] for o in cls2objindex[cls_]]
		encoder_input_ids.append(self.tokenizer.eos_token_id)
		encoder_img_mask.append(0)
		encoder_cls.append(0)

		dim_shape = sum([r.shape[0] for r in rel_position])
		encoder_rel_position_np = np.ones((dim_shape + 1, dim_shape + 1), dtype=np.int64) * 54
		if not self.config.use_orginal_enc_pos_embs:
			accumulate_dim = 0
			rel_start_position = []
			for r in rel_position:
				encoder_rel_position_np[accumulate_dim: accumulate_dim + r.shape[0], accumulate_dim: accumulate_dim + r.shape[0]] = r
				rel_start_position.append(accumulate_dim)
				accumulate_dim += r.shape[0]
			encoder_rel_position_np[-1, -1] = 0

			for i, ri in enumerate(rel_position):
				for j, rj in enumerate(rel_position):
					if i == j: continue
					i_vis_end = len(cls2objindex[key_order[i]])
					j_vis_end = len(cls2objindex[key_order[j]])
					for i_index in range(ri.shape[0]):
						for j_index in range(rj.shape[0]):
							if i_index < i_vis_end and j_index < j_vis_end:
								encoder_rel_position_np[rel_start_position[i] + i_index, rel_start_position[j] + j_index] = 50
							elif i_index < i_vis_end and j_index >= j_vis_end:
								encoder_rel_position_np[rel_start_position[i] + i_index, rel_start_position[j] + j_index] = 51
							elif i_index >= i_vis_end and j_index < j_vis_end:
								encoder_rel_position_np[rel_start_position[i] + i_index, rel_start_position[j] + j_index] = 52
							elif i_index >= i_vis_end and j_index >= j_vis_end:
								encoder_rel_position_np[rel_start_position[i] + i_index, rel_start_position[j] + j_index] = 53

		obj_feature_np = np.zeros((len(encoder_input_ids), obj_features.shape[-1]), dtype=np.float32)
		obj_box_np = np.zeros((len(encoder_input_ids), obj_boxes.shape[-1]), dtype=np.float32)
		obj_index = 0
		for i, m in enumerate(encoder_img_mask):
			if m == 1:
				if img_order[obj_index] < obj_size:
					cur_index = img_order[obj_index]
					obj_feature_np[i] = obj_features[cur_index]
					obj_box_np[i] = obj_boxes[cur_index]
				else:
					cur_index = img_order[obj_index] - obj_size
					obj_feature_np[i] = copy_obj_features[cur_index]
					obj_box_np[i] = copy_obj_boxes[cur_index]
				obj_index += 1
		
		self.obj_cache[img_id] = {
			"encoder_rel_position": encoder_rel_position_np,
			"encoder_input_ids": np.array(encoder_input_ids, dtype=np.int64),
			"encoder_cls": np.array(encoder_cls, dtype=np.int64),
			"encoder_img_mask": np.array(encoder_img_mask, dtype=np.float32),
			"obj_feature_np": obj_feature_np,
			"obj_box_np": obj_box_np,
			"image_id": img_id
		}

	def __getitem__(self, index):
		if self.is_training:
			img_id, cap, gt = self._captions_reader[index]
		else:
			img_id = self._image_ids[index]
			if self._captions_reader is not None:
				gt = self._captions_reader.get_gt_by_image_id(img_id)
			else:
				gt = None

			cap = None

		if img_id not in self.obj_cache:
			self.process_obj(img_id)

		item = self.obj_cache[img_id]

		if gt is not None:
			item['gt'] = gt

		if cap is not None:
			if cap not in self.cap_cache:
				self.process_cap(img_id, cap)
			item['cap'] = self.cap_cache[img_id][cap]['input_ids']
			item['mention_flag'] = self.cap_cache[img_id][cap]['mention_flag']
			item['mention_flag'][:, -1] = 0 
		else:
			item['mention_flag'] = self.global_obj_cache[img_id]
			item['mention_flag'][:, -1] = 0

		return item


def data_wrapper(config, dataset):
	new_dataset = {'image_ids': [int(d['image_id']) for d in dataset]}
	new_dataset['gt'] = [d['gt'] for d in dataset]

	encoder_input_ids, encoder_mask = process_tensor([d['encoder_input_ids'] for d in dataset], 0, output_mask=True)
	encoder_img_mask = process_tensor([d['encoder_img_mask'] for d in dataset], 0)
	encoder_cls = process_tensor([d['encoder_cls'] for d in dataset], 0)
	obj_feature = process_tensor([d['obj_feature_np'] for d in dataset], 2048)
	obj_box = process_tensor([d['obj_box_np'] for d in dataset], 8)

	new_dataset['encoder_input_ids'] = encoder_input_ids
	new_dataset['encoder_mask'] = encoder_mask
	new_dataset['encoder_img_mask'] = encoder_img_mask
	new_dataset['encoder_obj_feature'] = obj_feature
	new_dataset['encoder_obj_box'] = obj_box
	new_dataset['encoder_cls'] = encoder_cls

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

	encoder_rel_position = np.zeros((batch_size, max_encoder_len, max_encoder_len), dtype=np.int64)
	for i, d in enumerate(dataset):
		encoder_rel_position[i, :d['encoder_rel_position'].shape[0], :d['encoder_rel_position'].shape[1]] = d['encoder_rel_position']
	new_dataset['encoder_rel_position'] = torch.from_numpy(encoder_rel_position)

	return new_dataset

def get_data_loader(config, dataset):
    collate_fn = lambda d: data_wrapper(config, d)
    return DataLoader(dataset, 
        batch_size=config.batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )
