from dataset.vocabulary import T5CopyVocabulary
from dataset.e2e_dataset import E2EDataset, get_data_loader
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
import random
from dataset.pymteval import BLEUScore, NISTScore
from dataset.diversity import distinct_n
from constraint import CBSConstraint
import json

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
    "--constraint-vocab",
    default=None,
    help="Path to load constraint vocab",
)
parser.add_argument(
    "--output-path",
    default=None,
    help="Path to save output captions",
)
group = parser.add_mutually_exclusive_group()
group.add_argument('--train', action='store_true')
group.add_argument('--validation', action='store_true')
group.add_argument('--test', action='store_true')

def run_eval(_C, model, eval_data_iter, tokenizer, copy_vocab, device, decode_constraint=None, constraint_vocab=None, output_path=None):
    model.eval()
    if decode_constraint is not None:
        assert constraint_vocab is not None
        constraint_vocab_dict = {}
        with open(constraint_vocab) as out:
            for line in out:
                line = line.strip()
                items = line.split('@')
                constraint_vocab_dict[items[0]] = items[1:]


    gt_cap, pred = [], []
    obj_coverage = [0, 0]
    with torch.no_grad():
        for batch in tqdm(eval_data_iter):
            for n in batch:
                if n not in ['gt', 'gt_mr', 'ins_id']:
                    batch[n] = batch[n].to(device)

            if decode_constraint is not None:
                constraint_dict = {}
                for id_, gt_mr in enumerate(batch['gt_mr']):
                    constraint_dict[id_] = []
                    for (mr, _) in gt_mr:
                        if mr in constraint_vocab_dict:
                            c = []
                            for fg_w in constraint_vocab_dict[mr]:
                                fg_index = copy_vocab.w_to_i[fg_w]
                                c.append(copy_vocab.token_fg_w[fg_index])
                            constraint_dict[id_].append(c)

                state_transform_list = []
                state_num_list = []
                for image_id in range(len(batch['gt_mr'])):
                    state_matrix, state_num  = decode_constraint.get_state_matrix(_C.vocab_size, constraint_dict[image_id], image_id)
                    state_transform_list.append(state_matrix)
                    state_num_list.append(state_num)
                max_size = max(state_num_list)
                state_transform_list = [s[:, :max_size, :max_size]for s in state_transform_list]
                state_transition = np.concatenate(state_transform_list, axis=0)
                state_transition = torch.from_numpy(state_transition).bool().to(device)
            else:
                state_transition = None
            
            outputs = model.search(
            	input_ids=batch['encoder_input_ids'], 
                attention_mask=batch['encoder_mask'],
                decoder_mention_flag=batch['mention_flag'],
                decoder_cls_on_input=batch['encoder_cls'],
                state_transition=state_transition,
            	num_beams=5,
            	length_penalty=1.0,
            	max_length=_C.max_generation_len,
            	min_length=2,
            	no_repeat_ngram_size=3,
            	early_stopping=True
            )

            if decode_constraint is not None:
                outputs = decode_constraint.select_state_func(outputs, [i for i in range(len(batch['gt_mr']))])

            dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in outputs]
            for ins_id, d, gt, gt_mr in zip(batch['ins_id'], dec, batch['gt'], batch['gt_mr']):
                gt_cap.append(gt)
                pred.append((ins_id, d))
                gt_count = 0
                lower_d = d.lower()
                for (fullname, g_class_name) in gt_mr:
                    gt_count += 1
                    cls_id = copy_vocab.word_to_category_id[fullname]

                    has_found = False
                    for (w, _) in copy_vocab.d_to_w_group[cls_id]:
                        if w.lower() in lower_d:
                            obj_coverage[0] += 1
                            has_found = True
                            break

                    # if not has_found:
                    #     print(d)
                    #     print(copy_vocab.d_to_w_group[cls_id])
                    #     print([gt_mr])
                    #     print("-------")
                obj_coverage[1] += gt_count

    for p in pred[:20]:
        print(p)

    if output_path is not None:
        output_list = []
        for _id, out in pred:
            output_list.append({"image_id": _id, "caption": out})
        with open(output_path, 'w') as out:
            out.write(json.dumps(output_list))

    pred = [p[1] for p in pred]
    gts = evaluation.PTBTokenizer.tokenize(gt_cap)
    gen = evaluation.PTBTokenizer.tokenize(pred)

    print("Object Coverage %.2f" % (100 * obj_coverage[0] / obj_coverage[1]))

    diversity_sen = [v[0].split() for (_, v) in gen.items()]
    print("Diversity-1 %.2f" % distinct_n(diversity_sen, 1))
    print("Diversity-2 %.2f" % distinct_n(diversity_sen, 2))

    bleu = BLEUScore()
    nist = NISTScore()
    for sents_ref, sent_sys in zip(gt_cap, pred):
        bleu.append(sent_sys, sents_ref)
        nist.append(sent_sys, sents_ref)
    print("NIST %.2f" % (nist.score()))
    print("BLEU %.2f" % (bleu.score() * 100))

    val_meteor, _ = evaluation.Meteor().compute_score(gts, gen)
    print('METEOR %.2f' % (val_meteor * 100))

    val_cider, individual_cider = evaluation.Cider().compute_score(gts, gen)
    print('CIDEr %.2f' % (val_cider))

    val_rouge, _ = evaluation.Rouge().compute_score(gts, gen)
    print('ROUGE_L %.2f' % (val_rouge * 100))

    metric_dict = {"CIDEr": {"entire": val_cider}}
    metric_dict.update({"METEOR": {"entire": val_meteor}})

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
    copy_vocab = T5CopyVocabulary(_C.copy_vocab_path, tokenizer, sep='@')
    lm = get_lm_representation(_C, tokenizer, copy_vocab)
    model = lm['t5']
    model = model.to(device)
    _C.vocab_size = model.config.vocab_size

    total_parameter_count = 0
    trainable_parameter_count = 0
    for p in model.parameters():
        total_parameter_count += p.numel()
        if p.requires_grad:
            trainable_parameter_count += p.numel()
    print('Total Parameter Count %d' % total_parameter_count)
    print('Trainable Parameter Count %d' % trainable_parameter_count)

    if len(_C.decode_constrain) > 0:
        decode_constraint = CBSConstraint(_C.decode_constrain, 2)
    else:
        decode_constraint = None

    if _A.train:
        train_data = E2EDataset(_C, _C.train_path, tokenizer, copy_vocab, is_training=True)
        train_data_loader = get_data_loader(train_data, _C.batch_size)
        train_iter = iter(train_data_loader)

    dev_data = E2EDataset(_C, _C.dev_path if (_A.validation or _A.train) else _C.test_path, tokenizer, copy_vocab)
    dev_data_loader = get_data_loader(dev_data, _C.batch_size)

    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    if _A.validation or _A.test:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'model-best.pth'))['model'], strict=False)
        else:
            model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'model-best.pth'), map_location=torch.device('cpu'))['model'], strict=False)

        run_eval(_C, model, dev_data_loader, tokenizer, copy_vocab, device, decode_constraint=decode_constraint, constraint_vocab=_A.constraint_vocab, output_path=_A.output_path)


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
                        if n not in ['gt', 'gt_mr', 'ins_id']:
                            batch[n] = batch[n].to(device)
                    # optimizer.zero_grad()
                    total_step += 1
                    outputs = model(
                        input_ids=batch['encoder_input_ids'], 
                        attention_mask=batch['encoder_mask'],
                        decoder_mention_flag=batch['mention_flag'],
                        decoder_cls_on_input=batch['encoder_cls'],
                        labels=batch['cap_decoder_input_ids']
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

            eval_result = run_eval(_C, model, dev_data_loader, tokenizer, copy_vocab, device)
            checkpoint_manager.step(eval_result["CIDEr"]["entire"])
