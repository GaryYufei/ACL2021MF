from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import torch.nn as nn
import torch
import numpy as np
import copy
from transformers.modeling_t5 import *
from transformers.file_utils import ModelOutput
from transformers.generation_utils import *
import string
import sys
from dataclasses import dataclass
from transformers.generation_beam_search import *

@dataclass
class Seq2SeqLMOutputMF(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_mention_flags: Optional[Tuple[torch.LongTensor]] = None
    repeat_mask: Optional[Tuple[torch.BoolTensor]] = None


class ConstrainedBeamSearchScorer(BeamScorer):

    def __init__(
        self,
        batch_size: int,
        max_length: int,
        num_beams: int,
        num_states: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.num_states = num_states
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep

        self._is_init = False
        self._state_beam_hyps = [
            [
                BeamHypotheses(
                    num_beams=self.num_beams,
                    max_length=self.max_length,
                    length_penalty=self.length_penalty,
                    early_stopping=self.do_early_stopping,
                )
                for _ in range(num_states)
            ]
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([[False for _ in range(num_states)] for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._state_beam_hyps)

        device = input_ids.device
        next_beam_scores = torch.ones((batch_size, self.num_states, self.num_beams), dtype=next_scores.dtype, device=device) * -1e9
        next_beam_tokens = torch.zeros((batch_size, self.num_states, self.num_beams), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.num_states, self.num_beams), dtype=next_indices.dtype, device=device)

        for batch_idx, state_beam_hyp in enumerate(self._state_beam_hyps):
            for state_idx, beam_hyp in enumerate(state_beam_hyp):

                if self._done[batch_idx, state_idx]:
                    assert (
                        len(beam_hyp) >= self.num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(self.num_beams)
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    # pad the batch
                    next_beam_scores[batch_idx, state_idx, :] = -1e9
                    next_beam_tokens[batch_idx, state_idx, :] = pad_token_id
                    next_beam_indices[batch_idx, state_idx, :] = 0
                    continue

                beam_index = batch_idx * self.num_states + state_idx
                # next tokens for this sentence
                beam_idx = 0
                for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(next_tokens[beam_index], next_scores[beam_index], next_indices[beam_index])
                ):
                    batch_beam_idx = batch_idx * self.num_beams * self.num_states + next_index
                    # add to generated hypotheses if end of sentence 
                    if (eos_token_id is not None) and (next_token.item() == eos_token_id) and (next_score.item() > -1e9):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        beam_hyp.add(
                            input_ids[batch_beam_idx].clone(),
                            next_score.item(),
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_beam_scores[batch_idx, state_idx, beam_idx] = next_score
                        next_beam_tokens[batch_idx, state_idx, beam_idx] = next_token
                        next_beam_indices[batch_idx, state_idx, beam_idx] = batch_beam_idx
                        beam_idx += 1

                    # once the beam for next step is full, don't add more tokens to it.
                    if beam_idx == self.num_beams:
                        break

                # if beam_idx < self.num_beams:
                #     raise ValueError(
                #         f"At most {self.num_beams} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                #     )

            for state_idx, beam_hyp in enumerate(state_beam_hyp):
                beam_index = batch_idx * self.num_states + state_idx
                # Check if we are done so that we can save a pad step if all(done)
                self._done[batch_idx, state_idx] = self._done[batch_idx, state_idx] or beam_hyp.is_done(
                    next_scores[beam_index].max().item(), cur_len
                )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )


    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        batch_size = len(self._state_beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, state_beam_hyp in enumerate(self._state_beam_hyps):
            for state_idx, beam_hyp in enumerate(state_beam_hyp):
                if self._done[batch_idx, state_idx]:
                    continue

                start_index = batch_idx * self.num_beams * self.num_states + state_idx * self.num_beams
                # need to add best num_beams hypotheses to generated hyps
                for beam_id in range(self.num_beams):
                    batch_beam_idx = start_index + beam_id
                    final_score = final_beam_scores[batch_beam_idx].item()
                    final_tokens = input_ids[batch_beam_idx]
                    beam_hyp.add(final_tokens, final_score)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_states * self.num_beam_hyps_to_keep)
        best = []

        # retrieve best hypotheses
        for i, state_beam_hyp in enumerate(self._state_beam_hyps):
            for ii, beam_hyp in enumerate(state_beam_hyp):
                sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
                for j in range(self.num_beam_hyps_to_keep):
                    best_hyp = sorted_hyps.pop()[1]
                    sent_lengths[self.num_beam_hyps_to_keep * self.num_states * i + ii * self.num_beam_hyps_to_keep + j] = len(best_hyp)
                    best.append(best_hyp)

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_states * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return decoded

class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False, is_bidirectional=False, use_mention_flag=False, mention_flag_num=3, use_orginal_enc_pos_embs=True, use_mf_scalar=False):
        super().__init__()
        self.is_bidirectional = is_bidirectional
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv
        self.use_mention_flag = use_mention_flag

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.has_mention_flag = False
        self.use_mf_scalar = use_mf_scalar
        if self.use_mention_flag and mention_flag_num > 0:
            if not self.use_mf_scalar:
                self.k_mention_flag = nn.Embedding(mention_flag_num, self.d_kv)
                self.v_mention_flag = nn.Embedding(mention_flag_num, self.d_kv)
            else:
                self.mention_flag_scalar = nn.Embedding(mention_flag_num, self.n_heads)

            self.has_mention_flag = True

        self.use_orginal_enc_pos_embs = use_orginal_enc_pos_embs
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, self.d_kv, self.pruned_heads)
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.d_kv * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.is_bidirectional,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(
        self,
        input,
        mention_flag=None,
        mf_embed=None,
        mask=None,
        kv=None,
        position_bias=None,
        encoder_relative_pos_index=None,
        past_key_value=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # past_key_value[0] is (bs, n_heads, q_len - 1, dim_per_head)
        bs, qlen, dim = input.size()

        if past_key_value is not None:
            assert self.is_decoder is True, "Encoder cannot cache past key value states"
            assert (
                len(past_key_value) == 2
            ), "past_key_value should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value)
            )
            real_qlen = qlen + past_key_value[0].shape[2] if query_length is None else query_length
        else:
            real_qlen = qlen

        if kv is None:
            klen = real_qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

        if kv is None:
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif past_key_value is None:
            k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if past_key_value is not None:
            if kv is None:
                k_, v_ = past_key_value
                k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
            else:
                k, v = past_key_value

        if self.is_decoder and use_cache is True:
            present_key_value_state = ((k, v),)
        else:
            present_key_value_state = (None,)

        # (bs, n_heads, qlen, klen)
        scores = torch.matmul(
            q, k.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", q, k), compatible with onnx op>9

        if self.use_mention_flag:
            assert mention_flag is not None
            if not self.use_mf_scalar:
                B, h, len_q, d_k = q.size()
                rel_q = q.transpose(1, 2).view(B * len_q, h, d_k)
                rel_k = self.k_mention_flag(mention_flag)
                rel_k = rel_k.view(B * len_q, -1, d_k)
                rel_w = torch.matmul(rel_q, rel_k.transpose(-2, -1))
                scores = scores + rel_w.view(B, len_q, h, -1).transpose(1, 2)
            else:
                mf_scalar = self.mention_flag_scalar(mention_flag).permute([0, 3, 1, 2])
                scores = scores + mf_scalar


        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_qlen, klen), device=scores.device, dtype=scores.dtype
                )
            else:
                if self.is_decoder or self.use_orginal_enc_pos_embs:
                    position_bias = self.compute_bias(real_qlen, klen)
                else:
                    position_bias = self.relative_attention_bias(encoder_relative_pos_index).permute([0, 3, 1, 2])

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -qlen:, :]

            if mask is not None:
                position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)

        scores += position_bias

        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        if self.use_mention_flag and (not self.use_mf_scalar):
            assert mention_flag is not None
            rel_v = self.v_mention_flag(mention_flag)
            v = v.unsqueeze(2) + rel_v.unsqueeze(1)
            context = torch.matmul(v.transpose(-2, -1), weights.unsqueeze(-1)).squeeze(-1)
        else:
            context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)

        
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)

        outputs = (context,) + present_key_value_state

        if output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)
        # if self.has_mention_flag:
        #     outputs = outputs + (mf_embed,)

        return outputs

class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, use_mention_flag=False, mention_flag_num=3, use_mf_scalar=False):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias, is_bidirectional=True, use_mention_flag=use_mention_flag, mention_flag_num=mention_flag_num, use_mf_scalar=use_mf_scalar
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        kv,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        mention_flag=None,
        mf_embed=None
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            norm_x,
            mask=attention_mask,
            kv=kv,
            mention_flag=mention_flag,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            mf_embed=mf_embed
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, use_orginal_enc_pos_embs=True):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias, is_bidirectional=not config.is_decoder, use_orginal_enc_pos_embs=use_orginal_enc_pos_embs
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_relative_pos_index=None,
        head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            norm_x,
            mask=attention_mask,
            position_bias=position_bias,
            encoder_relative_pos_index=encoder_relative_pos_index,
            head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, use_mention_flag=False, mention_flag_num=3, use_orginal_enc_pos_embs=True, use_mf_scalar=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias, use_orginal_enc_pos_embs=use_orginal_enc_pos_embs))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, has_relative_attention_bias=has_relative_attention_bias, use_mention_flag=use_mention_flag, mention_flag_num=mention_flag_num, use_mf_scalar=use_mf_scalar))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        mention_flag=None,
        mf_embed=None,
        encoder_relative_pos_index=None,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=False,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_values,
                "2 (past / key) for cross attention" if expected_num_past_key_values == 4 else "",
                len(past_key_value),
            )
            assert len(past_key_value) == expected_num_past_key_values, error_message

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            encoder_relative_pos_index=encoder_relative_pos_index,
            head_mask=head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                mention_flag=mention_flag,
                mf_embed=mf_embed
            )
            hidden_states = cross_attention_outputs[0]
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        outputs = (hidden_states,)

        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)

class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None, use_mention_flag=False, mention_flag_num=3, visual_enable=False, visual_input_dim=0, use_orginal_enc_pos_embs=True, use_mf_scalar=False):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0), use_mention_flag=use_mention_flag, mention_flag_num=mention_flag_num, use_orginal_enc_pos_embs=use_orginal_enc_pos_embs, use_mf_scalar=use_mf_scalar) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.visual_enable = visual_enable
        if self.visual_enable:
            assert visual_input_dim > 0
            self.visual_head = nn.Linear(visual_input_dim, config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_img_mask=None,
        encoder_obj_feature=None,
        encoder_obj_box=None,
        encoder_relative_pos_index=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mention_flag=None
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        if self.visual_enable:
            assert encoder_img_mask is not None and encoder_obj_feature is not None and encoder_obj_box is not None
            encoder_img_mask = encoder_img_mask.unsqueeze(-1)
            visual_embeds = torch.cat([encoder_obj_feature, encoder_obj_box], dim=2)
            visual_embeds = self.visual_head(visual_embeds)
            inputs_embeds = visual_embeds * encoder_img_mask + inputs_embeds * (1 - encoder_img_mask)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                self
            )

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        mf_embed = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_relative_pos_index=encoder_relative_pos_index,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                mention_flag=mention_flag,
                mf_embed=mf_embed
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[3 if output_attentions else 2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4 if i == 0 else 3],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class T5WithMF(T5PreTrainedModel):
    authorized_missing_keys = [r"encoder\.embed_tokens\.weight", r"decoder\.embed_tokens\.weight", r"lm_head\.weight", r"mention_flag", r"visual_head", r"visual_encoder"]

    def __init__(self, config, **model_args):
        super().__init__(config)
        self.local_config = model_args['local_config']
        self.copy_vocab = model_args['copy_vocab']
        self.attachable_index = model_args['attachable_index']
        self.fg_str_dict = model_args['fg_str_dict']

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared, visual_enable=self.local_config.enable_visual, visual_input_dim=self.local_config.roi_dim + self.local_config.box_dim, use_orginal_enc_pos_embs=self.local_config.use_orginal_enc_pos_embs)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared, use_mention_flag=self.local_config.use_mention_flag, mention_flag_num=self.local_config.mention_flag_state, use_mf_scalar=self.local_config.use_mf_scalar)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def set_fg2cls(self, fg2cls):
        assert self.local_config.use_pointer
        self.register_buffer("fg2cls", nn.Parameter(fg2cls, requires_grad=False))

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_img_mask=None,
        encoder_obj_feature=None,
        encoder_obj_box=None,
        encoder_relative_pos_index=None,
        decoder_copy_pos=None,
        decoder_concept_cls=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_mention_flag=None,
        decoder_copy_mention_flag=None,
        decoder_cls_on_input=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decoder_history_input_ids=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """

        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("lm_labels")
        if "decoder_past_key_value_states" in kwargs:
            warnings.warn(
                "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_value_states")
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_img_mask=encoder_img_mask,
                encoder_obj_feature=encoder_obj_feature,
                encoder_obj_box=encoder_obj_box,
                encoder_relative_pos_index=encoder_relative_pos_index,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        B = decoder_input_ids.size(0)
        repeat_mask = torch.zeros((B, self.config.vocab_size)).bool().to(decoder_input_ids.device)
        if not self.training:
            B_len = decoder_history_input_ids.size(1)
            decoder_history_input_ids = decoder_history_input_ids.detach().cpu().numpy().tolist()
            cur_decoder_mention_flag = decoder_mention_flag.clone().squeeze(1)
            d_cls = decoder_cls_on_input.detach().cpu().numpy().tolist()
            d_mf = cur_decoder_mention_flag.detach().cpu().tolist()
            batch_has_overlap = []
            batch_found_cls = []

            for i in range(B):
                available_cls = set()
                for cls_index, mf in zip(d_cls[i], d_mf[i]):
                    if mf == 1 or mf == 2: available_cls.add(cls_index)

                has_overlap = False
                leading_ch = set()
                for cls_ in available_cls:
                    if self.copy_vocab.token_class[cls_][0] not in leading_ch:
                        leading_ch.add(self.copy_vocab.token_class[cls_][0])
                    else:
                        has_overlap = True
                batch_has_overlap.append(has_overlap)

                has_repeat = False
                all_fgs = []
                min_len = {}
                for cls_index in available_cls:
                    all_fgs += [(fg_index, cls_index) for (_, fg_index) in self.copy_vocab.d_to_w_group[cls_index]] 
                    min_len[cls_index] = min([len(self.copy_vocab.token_fg_w[fg_index]) for (_, fg_index) in self.copy_vocab.d_to_w_group[cls_index]] )
                all_fgs = sorted(all_fgs, key=lambda x: (min_len[x[1]], len(self.copy_vocab.token_fg_w[x[0]])), reverse=True)
                matched_position = []
                found_cls = set()
                for (fg_index, cls_index) in all_fgs:
                    s1 = self.fg_str_dict[fg_index]
                    fg_ch_list = self.copy_vocab.token_fg_w[fg_index]
                    for ch_idx, first_ch in enumerate(decoder_history_input_ids[i]):
                        if ch_idx in matched_position: continue
                        if first_ch == fg_ch_list[0]:
                            s2 = '&'.join([str(f) for f in decoder_history_input_ids[i][ch_idx: ch_idx + len(fg_ch_list)]])
                            if s1 == s2:
                                if not has_overlap:
                                    if cls_index not in found_cls:
                                        found_cls.add(cls_index)
                                        matched_position += [i for i in range(ch_idx, ch_idx + len(fg_ch_list))]
                                    else:
                                        has_repeat = True
                                else:
                                    if ch_idx + len(fg_ch_list) < B_len and decoder_history_input_ids[i][ch_idx + len(fg_ch_list)] not in self.attachable_index:
                                        if cls_index not in found_cls:
                                            found_cls.add(cls_index)
                                            matched_position += [i for i in range(ch_idx, ch_idx + len(fg_ch_list))]
                                        else:
                                            has_repeat = True
                                    elif ch_idx + len(fg_ch_list) == B_len:
                                        matched_position += [i for i in range(ch_idx, ch_idx + len(fg_ch_list))]

                        if has_repeat: break

                    if has_repeat: break

                batch_found_cls.append(found_cls)

                if self.local_config.decode_constrain == 'GBS':
                    if has_repeat:
                        # print([self.fg_str_dict[fg_index] for (_, fg_index) in self.copy_vocab.d_to_w_group[cls_index]])
                        # print(decoder_history_input_ids[i])
                        # print(has_overlap)
                        # print("------------")
                        repeat_mask[i,:] = True
                    else:
                        banned_first_word = set()
                        exclude_ban_word = set()
                        for cls_index in available_cls:
                            if cls_index not in found_cls:
                                for (_, fg_index) in self.copy_vocab.d_to_w_group[cls_index]:
                                    for ch in self.copy_vocab.token_fg_w[fg_index]:
                                        exclude_ban_word.add(ch)

                        for cls_index in found_cls:
                            for (_, fg_index) in self.copy_vocab.d_to_w_group[cls_index]:
                                ban_ch = self.copy_vocab.token_fg_w[fg_index][0]
                                if ban_ch not in exclude_ban_word:
                                    banned_first_word.add(ban_ch)
                        for wid in banned_first_word:
                            repeat_mask[i, wid] = True


            if self.local_config.use_mention_flag:
                if not self.local_config.static_mf:
                    for i in range(B):
                        available_cls = set()
                        for cls_index, mf in zip(d_cls[i], d_mf[i]):
                            if mf == 1: available_cls.add(cls_index)
                        has_overlap = batch_has_overlap[i]

                        for cls_index in available_cls:                 
                            state_number = 1
                            for (_, fg_index) in self.copy_vocab.d_to_w_group[cls_index]:
                                s1 = self.fg_str_dict[fg_index]
                                fg_ch_list = self.copy_vocab.token_fg_w[fg_index]
                                for ch_idx, first_ch in enumerate(decoder_history_input_ids[i]):
                                    if first_ch == fg_ch_list[0]:
                                        s2 = '&'.join([str(f) for f in decoder_history_input_ids[i][ch_idx: ch_idx + len(fg_ch_list)]])
                                        if s1 == s2:
                                            if not has_overlap:
                                                state_number = 2
                                                break
                                            else:
                                                if ch_idx + len(fg_ch_list) < B_len and decoder_history_input_ids[i][ch_idx + len(fg_ch_list)] not in self.attachable_index:
                                                    state_number = 2
                                                    break

                                if state_number == 2: break
                                                        
                            if state_number == 2:
                                cur_decoder_mention_flag[i][decoder_cls_on_input[i] == cls_index] = state_number
                            
                decoder_mention_flag = cur_decoder_mention_flag.unsqueeze(1)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mention_flag=decoder_mention_flag
        )

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutputMF(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            decoder_mention_flags=decoder_mention_flag,
            repeat_mask=repeat_mask

        )

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        for kwarg in ['decoder_copy_pos', 'decoder_concept_cls', 'decoder_mention_flag', 'decoder_copy_mention_flag', 'decoder_cls_on_input', 'encoder_img_mask']: 
            if kwarg in model_kwargs:
                model_kwargs[kwarg] = model_kwargs[kwarg].index_select(0, expanded_return_idx)
        
        return input_ids, model_kwargs

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        model_kwargs["decoder_mention_flag"] = outputs.decoder_mention_flags

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):


        # cut decoder_input_ids if past is used
        history_ids = input_ids.clone()
        if past is not None:
            input_ids = input_ids[:, -1:]

        generation = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "decoder_history_input_ids": history_ids
        }

        for kwarg in ['decoder_copy_pos', 'decoder_concept_cls', 'decoder_mention_flag', 'decoder_copy_mention_flag', 'decoder_cls_on_input', 'encoder_img_mask', 'encoder_obj_feature', 'encoder_obj_box', 'encoder_relative_pos_index']:
            if kwarg in kwargs:
                generation[kwarg] = kwargs[kwarg]

        return generation

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    @torch.no_grad()
    def search(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        state_transition: Optional[torch.Tensor] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs
    ) -> torch.LongTensor:

        # set init values
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is None:
            # init `input_ids` with bos_token_id
            input_ids = self._prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

            # set input_ids as decoder_input_ids
            input_ids = self._prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id, **model_kwargs
            )

            if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
                raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

        # determine generation mode
        is_greedy_gen_mode = (num_beams == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and do_sample is False
        is_beam_sample_gen_mode = (num_beams > 1) and do_sample is True

        # set model_kwargs
        model_kwargs["use_cache"] = use_cache

        # get distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            eos_token_id=eos_token_id,
        )

        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # get probability distribution warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            # expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if state_transition is not None:
                beam_scorer = ConstrainedBeamSearchScorer(
                    batch_size=batch_size,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_states=state_transition.size(1),
                    device=self.device,
                    length_penalty=length_penalty,
                    do_early_stopping=early_stopping,
                    num_beam_hyps_to_keep=num_return_sequences,
                )
            else:
                beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=max_length,
                    num_beams=num_beams,
                    device=self.device,
                    length_penalty=length_penalty,
                    do_early_stopping=early_stopping,
                    num_beam_hyps_to_keep=num_return_sequences,
                )

            expand_size = num_beams
            if state_transition is not None:
                expand_size = expand_size * state_transition.size(1)

            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=expand_size, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                state_transition=state_transition,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            batch_size = input_ids.shape[0] * num_return_sequences

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # interleave with `num_beams * num_return_sequences`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        state_transition: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs
    ):

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        if state_transition is None:
            batch_size = len(beam_scorer._beam_hyps)
        else:
            batch_size = len(beam_scorer._state_beam_hyps)

        num_beams = beam_scorer.num_beams

        use_constrained_decoding = state_transition is not None

        num_states = 1
        if state_transition is not None:
            assert state_transition.size(1) == state_transition.size(2), "num state wrong"
            num_states = state_transition.size(1)
            state_transition = state_transition.unsqueeze(3).expand((-1, -1, -1, num_beams, -1))

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * num_states * batch_size  == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        if not use_constrained_decoding:
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
            beam_scores[:, 1:] = -1e9
        else:
            beam_scores = torch.zeros((batch_size, num_states, num_beams), dtype=torch.float, device=input_ids.device)
            beam_scores[:, 1:] = -1e9
            beam_scores[:, 0, 1:] = -1e9

        beam_scores = beam_scores.view((-1,))

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # adjust tokens for Bart, *e.g.*
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            next_token_scores = logits_processor(input_ids, next_token_scores)

            vocab_size = next_token_scores.shape[-1]
            next_token_scores[outputs.repeat_mask] = -1e9

            if use_constrained_decoding:
                next_token_scores = next_token_scores.view(batch_size, num_states, num_beams, -1)
                beam_scores = beam_scores.view(batch_size, num_states, num_beams)
                
                constrained_beam_scores = torch.FloatTensor(batch_size, num_states, 2 * num_beams).to(next_token_scores.device)
                constrained_beam_indices = torch.LongTensor(batch_size, num_states, 2 * num_beams).to(next_token_scores.device)

                for i in range(num_states):
                    cloned_scores = next_token_scores.clone()

                    cloned_scores[~state_transition[:, :, i, :, :]] = -1e9
                    overall_scores = cloned_scores + beam_scores[:, :, :, None].expand_as(cloned_scores)
                    overall_scores = overall_scores.view(batch_size, -1)

                    state_beam_log_probs, state_beam_indices = torch.topk(
                        overall_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                    )

                    constrained_beam_scores[:, i, :] = state_beam_log_probs
                    constrained_beam_indices[:, i, :] = state_beam_indices


                next_token_scores, next_tokens = constrained_beam_scores.view(-1, 2 * num_beams), constrained_beam_indices.view(-1, 2 * num_beams)

            else:
                next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            
                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if model_kwargs["decoder_mention_flag"] is not None:
                model_kwargs["decoder_mention_flag"] =  model_kwargs["decoder_mention_flag"].index_select(0, beam_idx)

            if beam_scorer.is_done:
                break

        decoded = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )
        
        if use_constrained_decoding:
            decoded = decoded.view(batch_size, num_states, -1)

        return decoded


def create_embeds(model, target_vocab):
    vocab_emb = model.get_input_embeddings().weight.detach().cpu().numpy()
    vec_dim = vocab_emb.shape[1]
    w_list = [None for _ in range(len(target_vocab))]
    for index in target_vocab:
        t = target_vocab[index]
        vec = np.zeros((vec_dim,))
        for w in t:
            if w >= 0:
                vec += vocab_emb[w]
            else:
                vec += np.random.random((vec_dim, ))
        w_list[index] = vec / len(t)
    return torch.tensor(np.array(w_list)).float()


def get_lm_representation(config, tokenizer, copy_vocab=None):
    attachable_index = set()
    for index in range(tokenizer.sp_model.get_piece_size()):
        x = tokenizer.sp_model.IdToPiece(index)
        if (not x[0] == chr(9601)) and (not all([c in string.punctuation for c in x])):
            attachable_index.add(index)

    fg_str_dict = None
    if copy_vocab is not None:
        fg_str_dict = {}
        for fg_index in copy_vocab.token_fg_w:
            fg_ch_list = copy_vocab.token_fg_w[fg_index]
            s1 = '&'.join([str(f) for f in fg_ch_list])
            fg_str_dict[fg_index] = s1

    if config.do_pretrain_lm_init:
        model = T5WithMF.from_pretrained(config.lm_type, return_dict=True, cache_dir='.', local_config=config, copy_vocab=copy_vocab, attachable_index=attachable_index, fg_str_dict=fg_str_dict)
    else:
        lm_config = T5Config.from_pretrained(config.lm_type, return_dict=True, cache_dir='.')
        lm_config.num_layers = 3
        model = T5WithMF(lm_config, local_config=config, copy_vocab=copy_vocab, attachable_index=attachable_index, fg_str_dict=fg_str_dict)

    if config.freeze_param and config.do_pretrain_lm_init:
        for p in model.shared.parameters():
            p.requires_grad = False

        for p in model.lm_head.parameters():
            p.requires_grad = False

        for dec_block in model.decoder.block:
            for p in dec_block.layer[0].parameters():
                p.requires_grad = False
            for p in dec_block.layer[-1].parameters():
                p.requires_grad = False

    enc_block = model.encoder.block[0]
    if config.use_orginal_enc_pos_embs:
        if config.freeze_enc_pos_param:
            # forzen the position embeddings
            for p in enc_block.layer[0].SelfAttention.relative_attention_bias.parameters():
                p.requires_grad = False
    else:
        assert config.relative_pos_num > 20, "new relative pos embeds should be positive"
        new_relative_attention_bias = np.empty((config.relative_pos_num, model.config.num_heads), dtype=np.float32)
        old_relative_attention_bias_np = enc_block.layer[0].SelfAttention.relative_attention_bias.weight.detach().cpu().numpy()
        for i in range(32):
            new_relative_attention_bias[i] = old_relative_attention_bias_np[i]
        enc_block.layer[0].SelfAttention.relative_attention_bias = nn.Embedding.from_pretrained(torch.tensor(new_relative_attention_bias), freeze=False)

    return {"t5": model, "attachable_index": attachable_index}
