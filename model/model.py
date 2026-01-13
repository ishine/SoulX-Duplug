import torch, torchaudio
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

import pytorch_lightning as pl
from tqdm import tqdm
import re
import math
import os, sys
import peft
from peft import LoraConfig, get_peft_model
import copy
import time

from model.glm_4_voice.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import WhisperFeatureExtractor


class EncoderProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_embed_dim = config.audio_embed_dim
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.audio_embed_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(2048, self.llm_dim)

    def forward(self, x):
        x = x.contiguous()  # (batch, seq_len, dim)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


class Dual_ASR_Model_duplex(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.asr_eos_loss_rate = config.asr_eos_loss_rate
        self.asr_eos_token_id = config.asr_eos_token_id
        self.lm_vocab_size = config.lm_vocab_size
        self.best_val_acc = 0.0
        self.save_hyperparameters(self.config)

        self.sampling_rate = 16000
        self.token_samples = int(0.08 * self.sampling_rate)
        self._resample_buffer: dict[int, torchaudio.transforms.Resample] = {}

        self.glm_tokenizer = WhisperVQEncoder.from_pretrained(config.glm_tokenizer_path)
        for name, param in self.glm_tokenizer.named_parameters():
            param.requires_grad = False
        self.glm_tokenizer.eval()

        if self.config.enable_projector:
            if self.global_rank == 0:
                print(f"setting up audio projector...")
            self.audio_projector = EncoderProjector(config)
            if self.config.freeze_projector:
                if self.global_rank == 0:
                    print(f"freeze audio projector...")
                for name, param in self.audio_projector.named_parameters():
                    param.requires_grad = False
                self.audio_projector.eval()
        else:
            self.audio_projector = None

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(config.model_name)

        for name, param in self.llm.named_parameters():
            param.requires_grad = True
        self.llm.train()

        if config.init_ckpt_path:
            print(f"loading state dict from {config.init_ckpt_path}...")
            checkpoint = torch.load(
                config.init_ckpt_path,
                map_location=torch.device("cpu"),
                weights_only=True,
            )
            state_dict = (
                checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            )
            self.load_state_dict(state_dict)

            del checkpoint

        if self.config.embed_only:
            if self.global_rank == 0:
                print(f"only train partial embedding weights...")
            self.partial_freeze_weights(
                self.config.original_vocab_size, self.config.lm_vocab_size
            )

        if self.config.enable_lora:
            if self.global_rank == 0:
                print(f"setting up lora model...")
            peft_config = LoraConfig(
                task_type=self.config.lora_task_type,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
            )
            self.llm = get_peft_model(self.llm, peft_config)

            if config.init_ckpt_path_lora:
                print(f"loading state dict from {config.init_ckpt_path_lora}...")
                checkpoint = torch.load(
                    config.init_ckpt_path_lora,
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
                state_dict = (
                    checkpoint["state_dict"]
                    if "state_dict" in checkpoint
                    else checkpoint
                )
                self.load_state_dict(state_dict)

                del checkpoint

    def forward(self, batch):
        sequences, audio_masks, labels = batch

        if self.audio_projector:
            audio_tokens = sequences.clone()
            audio_tokens[audio_masks] -= self.config.added_audio_token_start
            audio_tokens[~audio_masks] = 0
            audio_embeds = self.glm_tokenizer.codebook(audio_tokens)
            audio_embeds = self.audio_projector(audio_embeds)

            sequences[audio_masks] = 0

            if hasattr(self.llm.model, "embed_tokens"):
                text_embeds = self.llm.model.embed_tokens(sequences)
            elif hasattr(self.llm.model.model, "embed_tokens"):
                text_embeds = self.llm.model.model.embed_tokens(sequences)
            else:
                text_embeds = self.llm.model.model.model.embed_tokens(sequences)

            audio_masks = audio_masks.unsqueeze(-1)
            inputs_embeds = audio_embeds * audio_masks + text_embeds * (~audio_masks)

            model_outputs = self.llm(inputs_embeds=inputs_embeds, labels=labels)
        else:
            model_outputs = self.llm(input_ids=sequences, labels=labels)

        return model_outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        audio_masks = batch["audio_masks"]
        label_ids = batch["label_ids"]

        label_asr_ids = batch["label_asr_ids"]
        label_state_ids = batch["label_state_ids"]
        label_action_ids = batch["label_action_ids"]

        label_speak_ids = batch["label_speak_ids"]
        label_interrupted_ids = batch["label_interrupted_ids"]
        label_interrupt_ids = batch["label_interrupt_ids"]
        label_backchannel_ids = batch["label_backchannel_ids"]
        label_text_ids = batch["label_text_ids"]
        label_eos_ids = batch["label_eos_ids"]

        label_idle_ids = batch["label_idle_ids"]
        label_nonidle_ids = batch["label_nonidle_ids"]

        model_outputs = self((input_ids, audio_masks, label_ids))

        x_ori = model_outputs.logits

        # asr_loss = F.cross_entropy(
        #     x_ori[:, :-1, :].reshape(-1, self.lm_vocab_size),
        #     label_asr_ids[:, 1:].reshape(-1),
        #     ignore_index=-100,
        # )

        text_loss = F.cross_entropy(
            x_ori[:, :-1, :].reshape(-1, self.lm_vocab_size),
            label_text_ids[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        eos_loss = F.cross_entropy(
            x_ori[:, :-1, :].reshape(-1, self.lm_vocab_size),
            label_eos_ids[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        asr_loss = (
            self.config.asr_eos_loss_rate * eos_loss
            + (1 - self.config.asr_eos_loss_rate) * text_loss
        )

        # state_loss = F.cross_entropy(
        #     x_ori[:, :-1, :].reshape(-1, self.lm_vocab_size),
        #     label_state_ids[:, 1:].reshape(-1),
        #     ignore_index=-100,
        # )

        idle_loss = F.cross_entropy(
            x_ori[:, :-1, :].reshape(-1, self.lm_vocab_size),
            label_idle_ids[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        nonidle_loss = F.cross_entropy(
            x_ori[:, :-1, :].reshape(-1, self.lm_vocab_size),
            label_nonidle_ids[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        state_loss = (
            self.config.idle_loss_rate * idle_loss
            + (1 - self.config.idle_loss_rate) * nonidle_loss
        )

        action_loss = F.cross_entropy(
            x_ori[:, :-1, :].reshape(-1, self.lm_vocab_size),
            label_action_ids[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        # loss = F.cross_entropy(
        #     x_ori[:, :-1, :].reshape(-1, self.lm_vocab_size),
        #     label_ids[:, 1:].reshape(-1),
        #     ignore_index=-100,
        # )

        loss = (
            self.config.asr_loss_rate * asr_loss
            + self.config.state_loss_rate * state_loss
            + self.config.action_loss_rate * action_loss
        )

        preds = torch.argmax(x_ori, -1)
        acc = self.compute_accuracy(
            preds.detach()[:, :-1], label_ids.detach()[:, 1:], ignore_label=-100
        )
        asr_acc = self.compute_accuracy(
            preds.detach()[:, :-1], label_asr_ids.detach()[:, 1:], ignore_label=-100
        )
        state_acc = self.compute_accuracy(
            preds.detach()[:, :-1], label_state_ids.detach()[:, 1:], ignore_label=-100
        )
        action_acc = self.compute_accuracy(
            preds.detach()[:, :-1], label_action_ids.detach()[:, 1:], ignore_label=-100
        )

        speak_acc = self.compute_accuracy(
            preds.detach()[:, :-1], label_speak_ids.detach()[:, 1:], ignore_label=-100
        )
        interrupted_acc = self.compute_accuracy(
            preds.detach()[:, :-1],
            label_interrupted_ids.detach()[:, 1:],
            ignore_label=-100,
        )
        interrupt_acc = self.compute_accuracy(
            preds.detach()[:, :-1],
            label_interrupt_ids.detach()[:, 1:],
            ignore_label=-100,
        )
        backchannel_acc = self.compute_accuracy(
            preds.detach()[:, :-1],
            label_backchannel_ids.detach()[:, 1:],
            ignore_label=-100,
        )

        self.log("train_loss", loss, prog_bar=True, logger=True, rank_zero_only=True)

        self.log("train_acc", acc, prog_bar=True, logger=True, rank_zero_only=True)

        self.log(
            "train_asr_acc", asr_acc, prog_bar=True, logger=True, rank_zero_only=True
        )
        self.log(
            "train_state_acc",
            state_acc,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.log(
            "train_action_acc",
            action_acc,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )

        self.log(
            "train_speak_acc",
            speak_acc,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.log(
            "train_interrupted_acc",
            interrupted_acc,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.log(
            "train_interrupt_acc",
            interrupt_acc,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.log(
            "train_backchannel_acc",
            backchannel_acc,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )

        return loss

    # def on_training_epoch_end(self):
    #     pass

    def on_validation_epoch_start(self):
        self.val_loss = []
        self.val_loss_asr = []
        self.val_loss_state = []
        self.val_loss_action = []
        self.val_acc = []
        self.val_acc_asr = []
        self.val_acc_state = []
        self.val_acc_action = []

        self.val_acc_speak = []
        self.val_acc_interrupted = []
        self.val_acc_interrupt = []
        self.val_acc_backchannel = []

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        audio_masks = batch["audio_masks"]
        label_ids = batch["label_ids"]

        label_asr_ids = batch["label_asr_ids"]
        label_state_ids = batch["label_state_ids"]
        label_action_ids = batch["label_action_ids"]

        label_speak_ids = batch["label_speak_ids"]
        label_interrupted_ids = batch["label_interrupted_ids"]
        label_interrupt_ids = batch["label_interrupt_ids"]
        label_backchannel_ids = batch["label_backchannel_ids"]

        model_outputs = self((input_ids, audio_masks, label_ids))

        x_ori = model_outputs.logits

        asr_loss = F.cross_entropy(
            x_ori[:, :-1, :].reshape(-1, self.lm_vocab_size),
            label_asr_ids[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        state_loss = F.cross_entropy(
            x_ori[:, :-1, :].reshape(-1, self.lm_vocab_size),
            label_state_ids[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        action_loss = F.cross_entropy(
            x_ori[:, :-1, :].reshape(-1, self.lm_vocab_size),
            label_action_ids[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        # loss = F.cross_entropy(
        #     x_ori[:, :-1, :].reshape(-1, self.lm_vocab_size),
        #     label_ids[:, 1:].reshape(-1),
        #     ignore_index=-100,
        # )

        loss = (
            self.config.asr_loss_rate * asr_loss
            + self.config.state_loss_rate * state_loss
            + self.config.action_loss_rate * action_loss
        )

        preds = torch.argmax(x_ori, -1)
        acc = self.compute_accuracy(
            preds.detach()[:, :-1], label_ids.detach()[:, 1:], ignore_label=-100
        )
        asr_acc = self.compute_accuracy(
            preds.detach()[:, :-1], label_asr_ids.detach()[:, 1:], ignore_label=-100
        )
        state_acc = self.compute_accuracy(
            preds.detach()[:, :-1], label_state_ids.detach()[:, 1:], ignore_label=-100
        )
        action_acc = self.compute_accuracy(
            preds.detach()[:, :-1], label_action_ids.detach()[:, 1:], ignore_label=-100
        )

        speak_acc = self.compute_accuracy(
            preds.detach()[:, :-1], label_speak_ids.detach()[:, 1:], ignore_label=-100
        )
        interrupted_acc = self.compute_accuracy(
            preds.detach()[:, :-1],
            label_interrupted_ids.detach()[:, 1:],
            ignore_label=-100,
        )
        interrupt_acc = self.compute_accuracy(
            preds.detach()[:, :-1],
            label_interrupt_ids.detach()[:, 1:],
            ignore_label=-100,
        )
        backchannel_acc = self.compute_accuracy(
            preds.detach()[:, :-1],
            label_backchannel_ids.detach()[:, 1:],
            ignore_label=-100,
        )

        self.val_loss.append(loss)
        self.val_loss_asr.append(asr_loss)
        self.val_loss_state.append(state_loss)
        self.val_loss_action.append(action_loss)
        self.val_acc.append(acc)
        self.val_acc_asr.append(asr_acc)
        self.val_acc_state.append(state_acc)
        self.val_acc_action.append(action_acc)

        self.val_acc_speak.append(speak_acc)
        self.val_acc_interrupted.append(interrupted_acc)
        self.val_acc_interrupt.append(interrupt_acc)
        self.val_acc_backchannel.append(backchannel_acc)

    def on_validation_epoch_end(self):
        avg_loss = torch.nanmean(torch.stack(self.val_loss))
        avg_loss_asr = torch.nanmean(torch.stack(self.val_loss_asr))
        avg_loss_state = torch.nanmean(torch.stack(self.val_loss_state))
        avg_loss_action = torch.nanmean(torch.stack(self.val_loss_action))

        # avg_acc = torch.nanmean(torch.stack(self.val_acc))
        avg_acc_asr = torch.nanmean(torch.stack(self.val_acc_asr))
        avg_acc_state = torch.nanmean(torch.stack(self.val_acc_state))
        avg_acc_action = torch.nanmean(torch.stack(self.val_acc_action))
        avg_acc = (
            self.config.asr_loss_rate * avg_acc_asr
            + self.config.state_loss_rate * avg_acc_state
            + self.config.action_loss_rate * avg_acc_action
        )

        avg_acc_speak = torch.nanmean(torch.stack(self.val_acc_speak))
        avg_acc_interrupted = torch.nanmean(torch.stack(self.val_acc_interrupted))
        avg_acc_interrupt = torch.nanmean(torch.stack(self.val_acc_interrupt))
        avg_acc_backchannel = torch.nanmean(torch.stack(self.val_acc_backchannel))

        self.log(f"val_loss", avg_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            f"val_loss_asr", avg_loss_asr, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            f"val_loss_state",
            avg_loss_state,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"val_loss_action",
            avg_loss_action,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(f"val_acc", avg_acc, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            f"val_acc_asr", avg_acc_asr, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            f"val_acc_state", avg_acc_state, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            f"val_acc_action",
            avg_acc_action,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            f"val_acc_speak",
            avg_acc_speak,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"val_acc_interrupted",
            avg_acc_interrupted,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"val_acc_interrupt",
            avg_acc_interrupt,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"val_acc_backchannel",
            avg_acc_backchannel,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.best_val_acc = max(self.best_val_acc, avg_acc)
        self.log(
            f"best_val_acc",
            self.best_val_acc,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    # def configure_optimizers(self):
    #     optimizer = optim.AdamW(
    #         self.parameters(),
    #         lr=self.config.learning_rate,
    #         weight_decay=self.config.weight_decay,
    #         betas=self.config.betas,
    #         eps=self.config.eps,
    #     )

    #     scheduler_dict = {
    #         "scheduler": WarmupAnnealSteps(
    #             optimizer,
    #             warmup_step=self.config.warmup_steps,
    #             anneal_steps=[self.config.anneal_steps],
    #             anneal_rate=self.config.anneal_rate,
    #             final_lr=self.config.min_lr,
    #         ),
    #         "interval": "step",
    #         "frequency": 1,
    #     }

    #     return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    @torch.no_grad()
    def duplex_generate(self, in_wav_path):
        # TODO: set this in config
        device = "cuda"
        chunk_size_small = 160
        chunk_size_big = 960
        chunk_token_len_small = chunk_size_small // 80
        chunk_token_len_big = chunk_size_big // 80

        assert os.path.exists(in_wav_path)
        wav, sr = torchaudio.load(in_wav_path)

        if sr != 16000:
            if sr not in self._resample_buffer:
                self._resample_buffer[sr] = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=16000
                )
            wav = self._resample_buffer[sr](wav).squeeze()
            sr = 16000
        else:
            wav = wav.squeeze()

        # empty_wav = torch.zeros([16000])
        # wav = torch.cat((empty_wav, wav), dim=0)

        sampling_rate = 16000  # Fixed at 16kHz
        segment_duration = 0.16
        lookback_duration = 0.96
        lookahead_duration = 0.04
        valid_samples_per_segment = int(segment_duration * sampling_rate)
        lookback_samples_per_segment = int(lookback_duration * sampling_rate)
        lookahead_samples_per_segment = int(lookahead_duration * sampling_rate)

        input_text_tokens = torch.tensor(
            self.tokenizer.encode("<|task_duplex_predict|><|punctuation_off|>")
        ).to("cuda")

        audio_pad_token = torch.tensor(self.tokenizer.encode("<|padding|>")).to("cuda")
        audio_bos_token = torch.tensor(
            self.tokenizer.encode("<|begin_of_sentence|>")
        ).to("cuda")
        audio_eos_token = torch.tensor(self.tokenizer.encode("<|end_of_sentence|>")).to(
            "cuda"
        )

        if hasattr(self.llm.model, "embed_tokens"):
            text_embeds = self.llm.model.embed_tokens(input_text_tokens)
            audio_pad_embeds = self.llm.model.embed_tokens(audio_pad_token)
            audio_bos_embeds = self.llm.model.embed_tokens(audio_bos_token)
            audio_eos_embeds = self.llm.model.embed_tokens(audio_eos_token)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            text_embeds = self.llm.model.model.embed_tokens(input_text_tokens)
            audio_pad_embeds = self.llm.model.model.embed_tokens(audio_pad_token)
            audio_bos_embeds = self.llm.model.model.embed_tokens(audio_bos_token)
            audio_eos_embeds = self.llm.model.model.embed_tokens(audio_eos_token)
        else:
            text_embeds = self.llm.model.model.model.embed_tokens(input_text_tokens)
            audio_pad_embeds = self.llm.model.model.model.embed_tokens(audio_pad_token)
            audio_bos_embeds = self.llm.model.model.model.embed_tokens(audio_bos_token)
            audio_eos_embeds = self.llm.model.model.model.embed_tokens(audio_eos_token)

        pooling_kernel_size = self.glm_tokenizer.config.pooling_kernel_size or 1
        stride = (
            self.glm_tokenizer.conv1.stride[0]
            * self.glm_tokenizer.conv2.stride[0]
            * pooling_kernel_size
            * self.feature_extractor.hop_length
        )

        input_embeds = text_embeds
        past_key_values = None
        asr_result = []
        accumulate_token = []
        accumulate_token_len = 0

        wav = wav.cpu().numpy()
        wav_duration = wav.shape[0] / sampling_rate * 1000

        for start_sample in range(0, wav.shape[0], valid_samples_per_segment):
            audio_segment = wav[
                max(0, start_sample - lookback_samples_per_segment) : start_sample
                + valid_samples_per_segment
                + lookahead_samples_per_segment
            ]

            start_index = (
                start_sample - max(0, start_sample - lookback_samples_per_segment)
            ) // self.token_samples

            end_index = min(
                start_index + 2, math.ceil(audio_segment.shape[0] / self.token_samples)
            )

            valid_range = (start_index, end_index)

            features = self.feature_extractor(
                [audio_segment],
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
                device=device,
                padding="longest",
                pad_to_multiple_of=stride,
            )
            features = features.to(device=device)
            outputs = self.glm_tokenizer(**features)
            speech_tokens = outputs.quantized_token_ids

            attention_mask = features.attention_mask[
                :,
                :: self.glm_tokenizer.conv1.stride[0]
                * self.glm_tokenizer.conv2.stride[0],
            ]
            attention_mask = attention_mask[
                :, :: self.glm_tokenizer.config.pooling_kernel_size
            ]
            assert attention_mask.shape == speech_tokens.shape
            assert len(speech_tokens) == 1

            speech_token = speech_tokens[0][attention_mask[0].bool()].tolist()
            audio_tokens = speech_token[valid_range[0] : valid_range[1]]

            # lm generate
            input_audio_tokens = torch.tensor(audio_tokens).to("cuda")

            audio_embeds = self.glm_tokenizer.codebook(input_audio_tokens)
            audio_embeds = self.audio_projector(audio_embeds)

            if audio_embeds.shape[0] != chunk_token_len_small:
                audio_embeds = torch.cat(
                    (
                        audio_embeds,
                        audio_pad_embeds.expand(
                            chunk_token_len_small - audio_embeds.shape[0], -1
                        ),
                    ),
                    dim=0,
                )

            input_embeds = torch.cat((input_embeds, audio_embeds), dim=0).unsqueeze(0)

            outputs = self.llm(
                inputs_embeds=input_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits[0]
            past_key_values = outputs.past_key_values
            pred = torch.argmax(logits, -1)[-1]
            pred_label = self.tokenizer.decode(pred)

            print(f"[{start_sample//16}-{start_sample//16+160}]: {pred_label}")

            if hasattr(self.llm.model, "embed_tokens"):
                input_embeds = self.llm.model.embed_tokens(pred.unsqueeze(0))
            elif hasattr(self.llm.model.model, "embed_tokens"):
                input_embeds = self.llm.model.model.embed_tokens(pred.unsqueeze(0))
            else:
                input_embeds = self.llm.model.model.model.embed_tokens(
                    pred.unsqueeze(0)
                )

            accumulate_token.extend(audio_tokens)
            accumulate_token_len += chunk_token_len_small
            if accumulate_token_len == chunk_token_len_big:
                asr_segment = []
                input_embeds = torch.cat(
                    (input_embeds, audio_bos_embeds), dim=0
                ).unsqueeze(0)

                for i in range(self.config.max_chunk_token_length):
                    outputs = self.llm(
                        inputs_embeds=input_embeds,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    logits = outputs.logits[0]
                    past_key_values = outputs.past_key_values
                    pred = torch.argmax(logits, -1)[-1]

                    if pred == self.asr_eos_token_id:
                        break
                    else:
                        asr_segment.append(pred)
                        if hasattr(self.llm.model, "embed_tokens"):
                            input_embeds = self.llm.model.embed_tokens(
                                pred.unsqueeze(0)
                            ).unsqueeze(0)
                        elif hasattr(self.llm.model.model, "embed_tokens"):
                            input_embeds = self.llm.model.model.embed_tokens(
                                pred.unsqueeze(0)
                            ).unsqueeze(0)
                        else:
                            input_embeds = self.llm.model.model.model.embed_tokens(
                                pred.unsqueeze(0)
                            ).unsqueeze(0)

                asr_segment_text = ""
                if asr_segment:
                    try:
                        asr_segment_text = self.tokenizer.decode(
                            torch.stack(asr_segment, dim=0), skip_special_tokens=True
                        ).strip()

                        if self.check_en(asr_segment_text):
                            asr_segment_text += " "
                    except:
                        asr_segment_text = ""

                    asr_result.append(asr_segment_text)

                print(f"[user segment]: {asr_segment_text}")

                accumulate_token_len = 0
                accumulate_token = []
                input_embeds = audio_eos_embeds

    def compute_accuracy(self, pad_outputs, pad_targets, ignore_label):
        mask = pad_targets != ignore_label
        numerator = torch.sum(
            pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
        )
        denominator = torch.sum(mask)
        return numerator.float() / denominator.float()

    def partial_freeze_weights(self, original_vocabsize, total_vocabsize):
        self.hook_handles = []

        if self.global_rank == 0:
            print(
                f"Only training partial embedding layer, from {original_vocabsize} to {total_vocabsize}"
            )

        trainable_range = (original_vocabsize, total_vocabsize)

        # Define a hook to zero out the gradient for weights outside the trainable range during the backward pass
        def zero_out_gradient(grad):
            grad[: trainable_range[0], :] = 0
            grad[trainable_range[1] + 1 :, :] = 0
            return grad

        # Freeze all layers first
        for param in self.llm.parameters():
            param.requires_grad = False

        # Assuming the output layer is `lm_head`
        for param in self.llm.lm_head.parameters():
            # Compute the standard deviation for He initialization
            std_dev = (2.0 / param.size(1)) ** 0.5

            # Initialize the specific rows with He initialization
            param[original_vocabsize:total_vocabsize] = (
                torch.randn((trainable_range[1] - trainable_range[0], param.size(1)))
                * std_dev
            )
            param.requires_grad = True
            # Register the hook on the weight tensor
            handle = param.register_hook(zero_out_gradient)
            self.hook_handles.append(handle)

        if hasattr(self.llm.model, "model") and hasattr(
            self.llm.model.model, "embed_tokens"
        ):
            embed_tokens_module = self.llm.model.model.embed_tokens
        elif hasattr(self.llm.model, "embed_tokens"):
            embed_tokens_module = self.llm.model.embed_tokens
        else:
            raise AttributeError("Cannot find embed_tokens in self.llm.model")

        # For non-tied embedding layers, both the two embedding layers need to be hooked
        if self.llm.lm_head.weight.data_ptr() != embed_tokens_module.weight.data_ptr():
            for param in embed_tokens_module.parameters():
                std_dev = (2.0 / param.size(1)) ** 0.5
                param[original_vocabsize:total_vocabsize] = (
                    torch.randn(
                        (trainable_range[1] - trainable_range[0], param.size(1))
                    )
                    * std_dev
                )
                param.requires_grad = True
                handle = param.register_hook(zero_out_gradient)
                self.hook_handles.append(handle)

    def check_en(self, text):
        symbol_pattern = re.compile(
            r"[\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"
            r"\u2000-\u206F"
            r"\u3000-\u303F"
            r"\uFF00-\uFFEF]"
        )

        for char in reversed(text):
            if char.isdigit() or symbol_pattern.match(char):
                continue
            if char >= "\u4e00" and char <= "\u9fff":  # is chinese
                return False
            else:
                return True

        return True

    def repetition_penalty(self, logits, generated_ids, repetition_penalty):
        """
        Apply repetition penalty to the logits.
        """
        if repetition_penalty == 1.0:
            return logits

        # Gather the logits for generated_ids
        score = torch.gather(logits, -1, generated_ids.unsqueeze(0))

        # Apply penalty
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )

        # Scatter the updated scores back into logits
        logits.scatter_(-1, generated_ids.unsqueeze(0), score)

        return logits
