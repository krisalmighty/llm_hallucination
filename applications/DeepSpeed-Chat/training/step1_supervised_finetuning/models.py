import torch
from transformers import (
    AutoModelForCausalLM, LlamaForCausalLM)
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
class LlamaForCausalLMVertA(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
    # class LlamaForCausalLM(LlamaPreTrainedModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
#        print(outputs[0].shape)
#        print(outputs[1].shape)
#        print(outputs[2].shape)
#       class BaseModelOutputWithPast(ModelOutput):
#       Sequence of hidden-states at the output of the last layer of the model.
        q = outputs["last_hidden_state"]
        D = q.shape[-1]
        # Hidden-states of the model at the output of each layer plus the optional initial embedding outputs
        kv = torch.cat([item.unsqueeze(2) for item in outputs["hidden_states"][:]], 2)
#        print("kv shape:", kv.shape)
#        print("q shape:", q.shape)
#       k  = 32 here
        # q: B * L * D
        # kv: B * L * K * D -> B * L * D * K
        # B * L * K
        # Compute the dot product of q and kv.transpose(-1, -2)
        # and scale it by 1 / sqrt(D)
        q_a = q / q.norm(dim=2, keepdim=True)
        kv_a = kv / kv.norm(dim=2, keepdim=True)

        scores = torch.einsum("ijk,ijkl->ijl", q_a, kv_a.transpose(-1, -2)) * 0.4

        # Apply softmax along the last dimension to get the attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        # add dropout !!!!
        # print("attn_weights:")
        # print(attn_weights)
        # Multiply the attention weights with kv to get the output
        attn_output = torch.einsum("ijl,ijlk->ijk", attn_weights, kv)
#        ori_hidden_states = outputs["hidden_states"]

        hidden_states = attn_output #+ q 

#        print("last_hidden_states:")
#        print(outputs["last_hidden_state"].shape)
#        print("hidden_states:")
#        print(hidden_states.shape)
#        hidden_states = outputs[0]
#        hidden_states = outputs[0]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class LlamaForCausalLMVertSelfCTSA(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.margin = 0.5
    def build_mask_matrix(self, seqlen, valid_len_list):
        '''
            (1) if a sequence of length 4 contains zero padding token (i.e., the valid length is 4),
                then the loss padding matrix looks like
                     [0., 1., 1., 1.],
                     [1., 0., 1., 1.],
                     [1., 1., 0., 1.],
                     [1., 1., 1., 0.]

            (2) if a sequence of length 4 contains 1 padding token (i.e., the valid length is 3),
                then the loss padding matrix looks like
                     [0., 1., 1., 0.],
                     [1., 0., 1., 0.],
                     [1., 1., 0., 0.],
                     [0., 0., 0., 0.]
        '''
        res_list = []
        base_mask = torch.ones(seqlen, seqlen) - torch.eye(seqlen, seqlen)
        base_mask = base_mask.type(torch.FloatTensor)
        bsz = len(valid_len_list)
        for i in range(bsz):
            one_base_mask = base_mask.clone()
            one_valid_len = valid_len_list[i]
            one_base_mask[:,one_valid_len:] = 0.
            one_base_mask[one_valid_len:, :] = 0.
            res_list.append(one_base_mask)
        res_mask = torch.stack(res_list, dim = 0)#torch.FloatTensor(res_list)
        #print (res_mask)
        assert res_mask.size() == torch.Size([bsz, seqlen, seqlen])
        return res_mask

    def contrastive_loss(self, score_matrix, input_ids):
        '''
           score_matrix: bsz x seqlen x seqlen
           input_ids: bsz x seqlen
        '''
        bsz, seqlen, _ = score_matrix.size()
        gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2) # bsz x seqlen
        gold_score = torch.unsqueeze(gold_score, -1)
        assert gold_score.size() == torch.Size([bsz, seqlen, 1])
        difference_matrix = gold_score - score_matrix
        assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
        loss_matrix = self.margin - difference_matrix # bsz x seqlen x seqlen  margin 越小  最后的层间difference越大
        loss_matrix = torch.nn.functional.relu(loss_matrix)

        ### input mask
        input_mask = torch.ones_like(input_ids).type(torch.FloatTensor)
        if loss_matrix.is_cuda:
            input_mask = input_mask.cuda(loss_matrix.get_device())
        input_mask = input_mask.masked_fill(input_ids.eq(0), 0.0)

        if loss_matrix.is_cuda:
            input_mask = input_mask.cuda(loss_matrix.get_device())

        valid_len_list = torch.sum(input_mask, dim = -1).tolist()
        loss_mask = self.build_mask_matrix(seqlen, [int(item) for item in valid_len_list])
        if score_matrix.is_cuda:
            loss_mask = loss_mask.cuda(score_matrix.get_device())
        masked_loss_matrix = loss_matrix * loss_mask

        loss_matrix = torch.sum(masked_loss_matrix, dim = -1)
        assert loss_matrix.size() == input_ids.size()
        loss_matrix = loss_matrix * input_mask
        cl_loss = torch.sum(loss_matrix) / torch.sum(loss_mask)
        return cl_loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        q = outputs["last_hidden_state"]
        D = q.shape[-1]
        kv = torch.cat([item.unsqueeze(2) for item in outputs["hidden_states"][:]], 2)
#        kv_cts = kv.squeeze() #torch.reshape(kv, (kv.shape[0] * kv.shape[1], kv.shape[1], kv.shape[2]))
#        kv_cts = kv_cts[:torch.sum(attention_mask.squeeze()).item(), :, :]
#        norm_rep = kv_cts / kv_cts.norm(dim=2, keepdim=True)
#        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2))         
#        cts_loss = self.contrastive_loss(cosine_scores, torch.ones(kv_cts.shape[0], kv_cts.shape[1]).cuda())
        #q: B * L * D
        #kv: B * L * K * D 
        # B * L * K
        # Compute the dot product of q and kv.transpose(-1, -2)
        # and scale it by 1 / sqrt(D)
        scores = torch.einsum("ijkl,ijml->ijkm", kv, kv) / math.sqrt(D)
#        print("scores:", scores.shape)
        # Apply softmax along the last dimension to get the attention weights
        attn_weights = torch.softmax(scores, dim=-1)
#        print("attn_weights:", attn_weights.shape)

#        print("attn_weights:")
#        print(attn_weights)
        # Multiply the attention weights with kv to get the output
        kv = torch.einsum("ijkm,ijml->ijkl", attn_weights, kv)

        scores = torch.einsum("ijkl,ijml->ijkm", kv, kv) / math.sqrt(D)

        attn_weights = torch.softmax(scores, dim=-1)

        attn_output = torch.einsum("ijkm,ijml->ijkl", attn_weights, kv)



        hidden_states = 0.1 * attn_output[:, :, -1, :] + 0.9 * q 


        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
#        print("cts_loss:")
#        print(cts_loss)
#        loss += 0.001 * cts_loss
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LlamaForCausalLMVertSelfCascade(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        q = outputs["last_hidden_state"]
        D = q.shape[-1]
        kv = torch.cat([item.unsqueeze(2) for item in outputs["hidden_states"][:]], 2)

        #q: B * L * D
        #kv: B * L * K * D 
        # B * L * K
        # Compute the dot product of q and kv.transpose(-1, -2)
        # and scale it by 1 / sqrt(D)
        scores = torch.einsum("ijkl,ijml->ijkm", kv, kv) / math.sqrt(D)
#        print("scores:", scores.shape)
        # Apply softmax along the last dimension to get the attention weights
        attn_weights = torch.softmax(scores, dim=-1)
#        print("attn_weights:", attn_weights.shape)

#        print("attn_weights:")
#        print(attn_weights)
        # Multiply the attention weights with kv to get the output
        attn_output = torch.einsum("ijkm,ijml->ijkl", attn_weights, kv)

#        scores = torch.einsum("ijkl,ijml->ijkm", kv, kv) / math.sqrt(D)

#        attn_weights = torch.softmax(scores, dim=-1)

#        attn_output = torch.einsum("ijkm,ijml->ijkl", attn_weights, kv)



        hidden_states = 0.1 * attn_output[:, :, 8, :] + 0.1 * attn_output[:, :, 16, :] + 0.1 * attn_output[:, :, 24, :] + 0.7 * q 

#        print("last_hidden_states:")
#        print(outputs["last_hidden_state"].shape)
#        print("hidden_states:")
#        print(hidden_states.shape)
#        hidden_states = outputs[0]
#        hidden_states = outputs[0]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LlamaForCausalLMVertSelfA(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        q = outputs["last_hidden_state"]
        D = q.shape[-1]
        kv = torch.cat([item.unsqueeze(2) for item in outputs["hidden_states"][:]], 2)
        kv_a = kv / kv.norm(dim=2, keepdim=True)

#        scores = torch.einsum("ijk,ijkl->ijl", q_a, kv_a.transpose(-1, -2)) / 0.1

        #q: B * L * D
        #kv: B * L * K * D 
        # B * L * K
        # Compute the dot product of q and kv.transpose(-1, -2)
        # and scale it by 1 / sqrt(D)
        
        scores = torch.einsum("ijkl,ijml->ijkm", kv_a, kv_a) / 0.1
        # math.sqrt(D)
#        print("scores:", scores.shape)
        # Apply softmax along the last dimension to get the attention weights
        attn_weights = torch.softmax(scores, dim=-1)
#        print("attn_weights:", attn_weights.shape)

#        print("attn_weights:")
#        print(attn_weights)
        # Multiply the attention weights with kv to get the output
        kv = torch.einsum("ijkm,ijml->ijkl", attn_weights, kv)
        kv_a = kv / kv.norm(dim=2, keepdim=True)

#        scores = torch.einsum("ijkl,ijml->ijkm", kv, kv) / math.sqrt(D)
        scores = torch.einsum("ijkl,ijml->ijkm", kv_a, kv_a) / 0.1

        attn_weights = torch.softmax(scores, dim=-1)

        attn_output = torch.einsum("ijkm,ijml->ijkl", attn_weights, kv)



        hidden_states = attn_output[:, :, -1, :]  #0.1 * attn_output[:, :, 16, :] + 0.1 * attn_output[:, :, 24, :] + 0.8 * q 

#        print("last_hidden_states:")
#        print(outputs["last_hidden_state"].shape)
#        print("hidden_states:")
#        print(hidden_states.shape)
#        hidden_states = outputs[0]
#        hidden_states = outputs[0]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LlamaForCausalLMVertSelfA_BK(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        q = outputs["last_hidden_state"]
        D = q.shape[-1]
        kv = torch.cat([item.unsqueeze(2) for item in outputs["hidden_states"][:]], 2)

        #q: B * L * D
        #kv: B * L * K * D 
        # B * L * K
        # Compute the dot product of q and kv.transpose(-1, -2)
        # and scale it by 1 / sqrt(D)
        
        scores = torch.einsum("ijkl,ijml->ijkm", kv, kv) / math.sqrt(D)
#        print("scores:", scores.shape)
        # Apply softmax along the last dimension to get the attention weights
        attn_weights = torch.softmax(scores, dim=-1)
#        print("attn_weights:", attn_weights.shape)

#        print("attn_weights:")
#        print(attn_weights)
        # Multiply the attention weights with kv to get the output
        kv = torch.einsum("ijkm,ijml->ijkl", attn_weights, kv)

        scores = torch.einsum("ijkl,ijml->ijkm", kv, kv) / math.sqrt(D)

        attn_weights = torch.softmax(scores, dim=-1)

        attn_output = torch.einsum("ijkm,ijml->ijkl", attn_weights, kv)



        hidden_states = 0.1 * attn_output[:, :, 16, :] + 0.1 * attn_output[:, :, 24, :] + 0.8 * q 

#        print("last_hidden_states:")
#        print(outputs["last_hidden_state"].shape)
#        print("hidden_states:")
#        print(hidden_states.shape)
#        hidden_states = outputs[0]
#        hidden_states = outputs[0]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

