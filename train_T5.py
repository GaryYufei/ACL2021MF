from dataset.vocabulary import T5CopyVocabulary
from dataset.dataset import CommonGenDataset, get_data_loader
import argparse
import torch
import torch.nn as nn
from config import Config
import numpy as np
from transformers import T5Tokenizer
from checkpointing import CheckpointManager
from t5 import get_lm_representation 
import utils
from tqdm import tqdm
import math
import os, sys
from speaksee import evaluation
import spacy
import random
from constraint import CBSConstraint
from dataset.diversity import distinct_n
import json

nlp = spacy.load("en_core_web_sm")
nlp.pipeline = [('tagger', nlp.tagger)]

def tokenize(_list):
    new_dict = {}
    for item in _list:
        if isinstance(item, list):
            new_sentence_list = []
            for sentence in item:
                a = ''
                for token in nlp(sentence):
                    a += token.text
                    a += ' '
                new_sentence_list.append(a.rstrip())
            new_dict[len(new_dict)] = new_sentence_list
        else:
            a = ''
            for token in nlp(item):
                a += token.text
                a += ' '
            new_dict[len(new_dict)] = [a]

    return new_dict

def get_coverage_score(gt_concepts, pred):
    covs = []
    total_cs, match_cs = 0, 0
    for cs, p in zip(gt_concepts, pred):
        p = p.lower()
        if p.endswith('.'):
            p = p[:-1]
            p = p.strip()
        cs = set(cs)
        lemmas = set()
        for token in nlp(p):
            lemmas.add(token.lemma_)
        match_cs += len(lemmas&cs)
        total_cs += len(cs)
        cov = len(lemmas&cs)/len(cs)
        covs.append(cov)
    return 100 * sum(covs) / len(covs), 100 * match_cs / total_cs

parser = argparse.ArgumentParser("Train a CommonGen T5")
parser.add_argument(
    "--config", required=True, help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
    "nesting) using a dot operator. The actual config will be updated and recorded in "
    "the serialization directory.",
)
parser.add_argument(
    "--serialization-dir",
    default=None,
    help="Path to a (non-existent) directory for serializing checkpoints and tensorboard logs.",
)
parser.add_argument(
    "--start-from-checkpoint",
    default=None,
    help="Path to load checkpoint and continue training [only supported for module_training].",
)
parser.add_argument(
    "--output-path",
    default=None,
    help="Path to save output captions",
)
parser.add_argument(
    "--seen-constraint-path",
    default=None,
    help="Path to novel constraints",
)
group = parser.add_mutually_exclusive_group()
group.add_argument('--train', action='store_true')
group.add_argument('--validation', action='store_true')
group.add_argument('--test', action='store_true')

def run_eval(_C, model, eval_data_iter, copy_vocab, tokenizer, device, decoder_start_token_id, only_test=False, decode_constraint=None, output_path=None, seen_constraint_path=None):
    model.eval()
    gts, pred, gt_concepts = [], [], []
    cls_recall = [0, 0]
    novel_cls_recall = [0, 0]
    seen_cls_recall = [0, 0]
    
    seen_constraint_list = []
    if seen_constraint_path is not None:
        with open(seen_constraint_path) as out:
            for l in out:
                l = l.strip()
                seen_constraint_list.append(l)

    with torch.no_grad():
        for batch in tqdm(eval_data_iter):
            for n in batch:
                if n not in ['gt', 'gt_concepts']:
                    batch[n] = batch[n].to(device)

            cls_used = []
            for i in range(batch['concept_cls'].size(0)):
                gt_cls = []
                for j in range(batch['concept_cls'].size(1)):
                    ix = batch['concept_cls'][i][j].item()
                    if ix > 0:
                        gt_cls.append(ix)
                cls_used.append(set(gt_cls))

            if decode_constraint is not None:
                constraint_dict = {}
                for i in range(batch['concept_cls'].size(0)):
                    constraint_dict[i] = []
                    for cls_index in cls_used[i]:
                        c = []
                        for (_, fg_idx) in copy_vocab.d_to_w_group[cls_index]:
                            c.append(copy_vocab.token_fg_w[fg_idx])
                        constraint_dict[i].append(c)

                state_transform_list = []
                state_num_list = []
                for i in range(batch['concept_cls'].size(0)):
                    state_matrix, state_num  = decode_constraint.get_state_matrix(_C.vocab_size, constraint_dict[i], i)
                    state_transform_list.append(state_matrix)
                    state_num_list.append(state_num)
                max_size = max(state_num_list)
                state_transform_list = [s[:, :max_size, :max_size]for s in state_transform_list]
                state_transition_np = np.concatenate(state_transform_list, axis=0)
                state_transition = torch.from_numpy(state_transition_np).bool().to(device)
            else:
                state_transition = None
            
            outputs = model.search(
            	input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask'], 
                decoder_copy_pos=batch['copy_pos'],
                decoder_concept_cls=batch['concept_cls'],
                decoder_copy_mention_flag=batch['copy_mention_flag'],
                decoder_mention_flag=batch['decoder_mention_flag'],
                decoder_cls_on_input=batch['cls_on_input'],
                state_transition=state_transition,
            	num_beams=5,
            	length_penalty=1.0,
            	max_length=25,
            	min_length=2,
            	no_repeat_ngram_size=3,
            	early_stopping=True,
                decoder_start_token_id=decoder_start_token_id
            )

            if decode_constraint is not None:
                outputs = decode_constraint.select_state_func(outputs, [i for i in range(batch['concept_cls'].size(0))])

            dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in outputs]
            for d, gt in zip(dec, batch['gt']):
                gts.append(gt)
                pred.append(d)
            gt_concepts += batch['gt_concepts']

            N, D = outputs.size()
            for i in range(N):
                gt_cls = cls_used[i]

                mention_cls = []
                if _C.use_pointer:
                    for j in range(D):
                        ix = outputs[i][j].item()
                        if ix >= _C.vocab_size:
                            ix = ix - _C.vocab_size
                            _cls = copy_vocab.i_to_cls[ix]
                            mention_cls.append(copy_vocab.id_to_category[_cls])
                else:
                    w_list = dec[i].split()
                    if w_list[-1].endswith('.'):
                        w_list[-1] = w_list[-1][:-1]
                    w_list = [w[:-2] if w.endswith("'s") else w for w in w_list]
                    w_list = [w[:-1] if w.endswith(",") else w for w in w_list]
                    for gt_c in gt_cls:
                        for (w, _) in copy_vocab.d_to_w_group[gt_c]:
                            if w in w_list:
                                mention_cls.append(gt_c)
                                break

                mention_cls = set(mention_cls)

                novel_gt = set([c for c in gt_cls if copy_vocab.id_to_category[c] not in seen_constraint_list])
                seen_gt = set([c for c in gt_cls if copy_vocab.id_to_category[c] in seen_constraint_list])

                novel_mention = set([c for c in mention_cls if copy_vocab.id_to_category[c] not in seen_constraint_list])
                seen_mention = set([c for c in mention_cls if copy_vocab.id_to_category[c] in seen_constraint_list])

                cls_recall[1] += len(gt_cls)
                cls_recall[0] += len(gt_cls & mention_cls)

                novel_cls_recall[1] += len(novel_gt)
                seen_cls_recall[1] += len(seen_gt)

                novel_cls_recall[0] += len(novel_gt & novel_mention)
                seen_cls_recall[0] += len(seen_gt & seen_mention)



                # if len(gt_cls - (gt_cls & mention_cls)) > 0 and only_test:
                #     remaining_cls = gt_cls - (gt_cls & mention_cls)
                #     print([copy_vocab.id_to_category[c] for c in gt_cls], [copy_vocab.id_to_category[c] for c in remaining_cls], dec[i])
                # print([copy_vocab.id_to_category[c] for c in gt_cls], dec[i])

    for p in pred[:20]:
        print(p)

    if output_path is not None:
        output_list = []
        for _id, out in enumerate(pred):
            output_list.append({"image_id": _id, "caption": out})
        with open(output_path, 'w') as out:
            out.write(json.dumps(output_list))

    gts = tokenize(gts)
    gen = tokenize(pred)

    coverage_score, overall_coverage = get_coverage_score(gt_concepts, pred)
    print("Coverage %.2f" % coverage_score)
    print("Macro Coverage %.2f" % overall_coverage)
    print("Token-Level Coverage %.2f" % (100 * cls_recall[0] / cls_recall[1]))
    if len(seen_constraint_list) > 0:
        print("Novel Token-Level Coverage %.2f" % (100 * novel_cls_recall[0] / novel_cls_recall[1]))
        print("Seen Token-Level Coverage %.2f" % (100 * seen_cls_recall[0] / seen_cls_recall[1]))


    diversity_sen = [v[0].split() for (_, v) in gen.items()]
    print("Diversity-1 %.2f" % distinct_n(diversity_sen, 1))
    print("Diversity-2 %.2f" % distinct_n(diversity_sen, 2))

    val_bleu, _ = evaluation.Bleu(n=4).compute_score(gts, gen)
    method = ['Blue_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
    metric_dict = {}
    for metric, score in zip(method, val_bleu):
        metric_dict['metric'] = {'entire': score * 100}
        print('%s %.2f' % (metric, score * 100))

    val_meteor, _ = evaluation.Meteor().compute_score(gts, gen)
    print('METEOR %.2f' % (val_meteor * 100))

    val_rouge, _ = evaluation.Rouge().compute_score(gts, gen)
    print('ROUGE_L %.2f' % (val_rouge * 100))

    val_cider, _ = evaluation.Cider().compute_score(gts, gen)
    print('CIDEr %.2f' % (val_cider * 100))

    val_spice, _ = evaluation.Spice().compute_score(gts, gen)
    print('SPICE %.2f' % (val_spice * 100))

    metric_dict.update({"CIDEr": {"entire": val_cider}, "ROUGE_L": {"entire": val_rouge}, "METEOR": {"entire": val_meteor}, "SPICE": {"entire": val_spice}})
    return metric_dict


if __name__ == "__main__":
    _A = parser.parse_args()

    _C = Config(_A.config, _A.config_override)

    np.random.seed(_C.random_seed)
    random.seed(_C.random_seed)
    torch.manual_seed(_C.random_seed)
    torch.cuda.manual_seed_all(_C.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = T5Tokenizer.from_pretrained(_C.lm_type, cache_dir='.')
    copy_vocab = T5CopyVocabulary(_C.copy_vocab_path, tokenizer)
    lm = get_lm_representation(_C, tokenizer, copy_vocab)
    model = lm['t5']
    model = model.to(device)
    _C.vocab_size = model.config.vocab_size

    if len(_C.decode_constrain) > 0:
        decode_constraint = CBSConstraint(_C.decode_constrain, 5)
    else:
        decode_constraint = None

    total_parameter_count = 0
    trainable_parameter_count = 0
    for p in model.parameters():
        total_parameter_count += p.numel()
        if p.requires_grad:
            trainable_parameter_count += p.numel()
    print('Total Parameter Count %d' % total_parameter_count)
    print('Trainable Parameter Count %d' % trainable_parameter_count)

    if _A.train:
        train_data = CommonGenDataset(_C, _C.train_path, tokenizer, copy_vocab, model.config.decoder_start_token_id, attachable_index=lm['attachable_index'], is_training=True)
        train_data_loader = get_data_loader(train_data, _C.batch_size)
        train_iter = iter(train_data_loader)

    dev_data = CommonGenDataset(_C, _C.dev_path if (_A.validation or _A.train) else _C.test_path, tokenizer, copy_vocab, model.config.decoder_start_token_id)
    dev_data_loader = get_data_loader(dev_data, _C.batch_size)

    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    if _A.validation or _A.test:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'model-best.pth'))['model'], strict=False)
        else:
            model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'model-best.pth'), map_location=torch.device('cpu'))['model'], strict=False)

        run_eval(_C, model, dev_data_loader, copy_vocab, tokenizer, device, model.config.decoder_start_token_id, only_test=True, decode_constraint=decode_constraint, output_path=_A.output_path, seen_constraint_path=_A.seen_constraint_path)


    if _A.train:
        _C.num_training_steps = len(train_iter) * _C.max_epoch / _C.gradient_accumulation_steps
        epoch_num = math.ceil(_C.num_training_steps / _C.checkpoint_every_step)

        checkpoint_manager = CheckpointManager(model, _A.serialization_dir, mode="max")
        optimizer = utils.build_optimizer(_C, model)

        os.makedirs(_A.serialization_dir, exist_ok=True)
        _C.dump(os.path.join(_A.serialization_dir, "config.yml"))

        eval_every = _C.checkpoint_every_step * _C.gradient_accumulation_steps
        total_step = 0

        for epoch in range(epoch_num):
            print('EPOCH %d / %d' % (epoch + 1, epoch_num))
            run_step = eval_every if total_step + eval_every < len(train_iter) * _C.max_epoch else  len(train_iter) * _C.max_epoch - total_step
            model.train()

            with tqdm(total=math.ceil(run_step / _C.gradient_accumulation_steps), file=sys.stdout) as pbar:
                for step in range(run_step):
                    try:
                        batch = next(train_iter)
                    except:
                        train_iter = iter(train_data_loader)
                        batch = next(train_iter)
                   
                    for n in batch:
                        if n not in ['gt', 'gt_concepts']:
                            batch[n] = batch[n].to(device)
                    total_step += 1
                    # optimizer.zero_grad()
                    outputs = model(
                        input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'], 
                        decoder_copy_pos=batch['copy_pos'],
                        decoder_concept_cls=batch['concept_cls'],
                        decoder_input_ids=batch['decoder_input_ids'],
                        decoder_attention_mask=batch['decoder_input_mask'],
                        decoder_copy_mention_flag=batch['copy_mention_flag'],
                        decoder_mention_flag=batch['decoder_mention_flag'],
                        decoder_cls_on_input=batch['cls_on_input'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss
                    loss = loss / _C.gradient_accumulation_steps
                    loss.backward()

                    if _C.grad_clip_value > 0:
                        torch.nn.utils.clip_grad_value_(model.parameters(), _C.grad_clip_value)
                    if (step + 1) % _C.gradient_accumulation_steps == 0:
                        optimizer.step()
                        if torch.cuda.is_initialized():
                            torch.cuda.synchronize()
                        pbar.set_description("loss %.2f" % (loss.item() * _C.gradient_accumulation_steps))
                        pbar.update(1)
                        optimizer.zero_grad()

            eval_result = run_eval(_C, model, dev_data_loader, copy_vocab, tokenizer, device, model.config.decoder_start_token_id)
            checkpoint_manager.step(eval_result["CIDEr"]["entire"])
