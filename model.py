import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import cross_entropy,softmax
from torch.distributions import Categorical
from transformers import BertModel, RobertaModel, GPT2Model, LlamaModel, AutoConfig, BertConfig
from typing import Optional, Tuple

from huggingface_hub import login




class Eyettention2(nn.Module):
	_keys_to_ignore_on_load_missing = [r"position_ids"]
	def __init__(self, cf):
		super(Eyettention2, self).__init__()
		self.cf = cf
		self.window_width = 1
		self.hidden_size = 128
		self.atten_type = cf["atten_type"]

		#encoder
		encoder_config = AutoConfig.from_pretrained(self.cf["model_pretrained"])
		encoder_config.output_hidden_states=True
		self.encoder_config = encoder_config
		 # initiate Bert with pre-trained weights
		print("keeping Bert with pre-trained weights")

		if self.cf["model_pretrained"].startswith('RoBERTa'):
			self.encoder = RobertaModel.from_pretrained(self.cf["model_pretrained"],
														config = encoder_config,
														add_pooling_layer = False)
			self.LayerNorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.layer_norm_eps)

		elif self.cf["model_pretrained"].startswith('bert') or 'chinese-roberta' in self.cf["model_pretrained"]:
			self.encoder = BertModel.from_pretrained(self.cf["model_pretrained"],
														config = encoder_config,
														add_pooling_layer = False)
			self.LayerNorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.layer_norm_eps)

		#freeze the parameters in Bert model
		for param in self.encoder.parameters():
			param.requires_grad = False

		self.embedding_dropout = nn.Dropout(0.4)
		self.encoder_lstm1 = nn.LSTM(input_size = encoder_config.hidden_size, hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm2 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm3 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm4 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm5 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm6 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm7 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm8 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)

		#decoder
		self.position_embeddings = nn.Embedding(encoder_config.max_position_embeddings, encoder_config.hidden_size)
		self.attn_position = nn.Linear(self.hidden_size, self.hidden_size+1) #acoount for the word length feature

		#initialize eight decoder cells
		self.decoder_cell1 = nn.LSTMCell(encoder_config.hidden_size+2, self.hidden_size) #acoount for fixiation duration and landing position features
		self.decoder_cell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell4 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell5 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell6 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell7 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell8 = nn.LSTMCell(self.hidden_size, self.hidden_size)

		#fixation postion decoder
		self.wordindx_decoder_dense1 = nn.Linear(self.hidden_size*2+1, 512)
		self.wordindx_decoder_dense2 = nn.Linear(512, 256)
		self.wordindx_decoder_dense3 = nn.Linear(256, 256)
		self.wordindx_decoder_dense4 = nn.Linear(256, 256)
		self.wordindx_decoder_dense5 = nn.Linear(256, self.cf["max_sn_len"]*2-3)

		#fixation duration decoder
		self.dur_decoder_dense1 = nn.Linear(self.hidden_size*2+2, 1024) # inlude the additional feature of fixation landing position
		self.dur_decoder_dense2 = nn.Linear(1024, 64)
		self.dur_decoder_dense3 = nn.Linear(64, 1)

		#landing position decoder
		self.landpos_decoder_dense1 = nn.Linear(self.hidden_size*2+1, 128)
		self.landpos_decoder_dense2 = nn.Linear(128, 128)
		self.landpos_decoder_dense3 = nn.Linear(128, 128)
		self.landpos_decoder_dense4 = nn.Linear(128, 128)
		self.landpos_decoder_dense5 = nn.Linear(128, 1)

		self.dropout_LSTM = nn.Dropout(0.2)
		self.dropout_dense = nn.Dropout(0.2)
		#for scanpath generation
		self.softmax = nn.Softmax(dim=1)


	def pool_subwords_to_word(self, subword_emb, word_ids_sn, target, pool_method='sum'):
		#try batching computing
		# Pool bert subwords back to word level
		merged_word_att = torch.empty(subword_emb.shape[0], 0, self.encoder_config.hidden_size).to(subword_emb.device)
		if target == 'sn':
			max_len = self.cf["max_sn_len"] #CLS and SEP included
		elif target == 'sp':
			max_len = self.cf["max_sp_len"] - 1 #do not account the 'SEP' token, since sp decoder does not pass it as input to predict the next fixation.

		for word_idx in range(max_len):
			word_mask = (word_ids_sn == word_idx).unsqueeze(2).repeat(1, 1, self.encoder_config.hidden_size)
			#pooling method -> sum
			if pool_method=='sum':
				pooled_word_emb = torch.sum(subword_emb * word_mask, 1).unsqueeze(1) #[batch, 1, 768]
			elif pool_method=='mean':
				pooled_word_emb = torch.mean(subword_emb * word_mask, 1).unsqueeze(1) #[batch, 1, 768]
			merged_word_att = torch.cat([merged_word_att, pooled_word_emb], dim=1)
		mask_word = torch.sum(merged_word_att, 2).bool()
		return merged_word_att, mask_word


	def encode(self, sn_emd, sn_mask, word_ids_sn, sn_word_len):
		outputs = self.encoder(input_ids=sn_emd, attention_mask=sn_mask)
		hidden_rep_orig = outputs.last_hidden_state
		if word_ids_sn != None:
			# Pool bert subwords back to word level for english corpus
			merged_word_att, sn_mask_word = self.pool_subwords_to_word(hidden_rep_orig,
																		word_ids_sn,
																		target='sn',
																		pool_method='sum')
		else:#no pooling for Chinese corpus
			merged_word_att, sn_mask_word = hidden_rep_orig, None

		hidden_rep = self.embedding_dropout(merged_word_att)
		#eight LSTM layers for encoder
		x, (hn, hc) = self.encoder_lstm1(hidden_rep, None) #first layer not use residual, since input dimension is different
		x, (hn, hc) = self.encoder_lstm2(self.dropout_LSTM(x), None)
		residual = x
		x, (hn, hc) = self.encoder_lstm3(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm4(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm5(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm6(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm7(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm8(self.dropout_LSTM(x), None)
		x = x + residual

		#concatenate with the word length feature
		x = torch.cat((x, sn_word_len[:, :, None]), dim=2)
		return x, sn_mask_word

	def location_prediction(self, sp_enc_out, word_enc_out, sp_pos, sn_mask, timestep):
		#predict fixation location
		# General Attention:
		# score(ht,hs) = (ht^T)(Wa)hs
		# hs is the output from encoder
		# ht is the previous hidden state from decoder
		# self.attn(o): [batch, step, units]
		attn_prod = torch.matmul(self.attn_position(sp_enc_out.unsqueeze(1)), word_enc_out.permute(0,2,1)) # [batch, 1, step]
		#local attention
		aligned_position = sp_pos[:, timestep]

		# Get window borders
		left = torch.where(aligned_position - self.window_width >= 0, (aligned_position - self.window_width), 0)
		right = torch.where(aligned_position + self.window_width <= self.cf["max_sn_len"]-1, aligned_position + self.window_width, self.cf["max_sn_len"]-1)

		#exclude padding tokens
		#only consider words in the window
		sen_seq = torch.arange(self.cf["max_sn_len"])[None,:].expand(sn_mask.shape[0],self.cf["max_sn_len"]).to(sn_mask.device)
		outside_win_mask = (sen_seq < left.unsqueeze(1)) +  (sen_seq > right.unsqueeze(1))
		attn_prod += (~sn_mask + outside_win_mask).unsqueeze(1) * -1e9
		#attn_prod += (torch.Tensor.bool(1-sn_mask_word) + outside_win_mask).unsqueeze(1) * -1e9
		att_weight = softmax(attn_prod, dim=2)             # [batch, 1, step]
		if self.atten_type == 'local-g': # local attention with Gaussian Distribution
			#gauss = lambda s: torch.exp(-torch.square(s - aligned_position.unsqueeze(1)) / (2 * torch.square(torch.tensor(self.window_width / 2))))
			gauss = lambda s: torch.exp(-torch.square(s - aligned_position.unsqueeze(1)) / (2 * torch.square(torch.tensor(self.window_width*self.cf["gaus_sd"]))))
			gauss_factor = gauss(sen_seq)
			att_weight = att_weight * gauss_factor.unsqueeze(1)

		#atten_weights_batch = torch.cat([atten_weights_batch, att_weight], dim=1)
		context = torch.matmul(att_weight, word_enc_out)    # [batch, 1, units]
		hc = torch.cat([context.squeeze(1),sp_enc_out],dim=1)      # [batch, units *2]

		hc = self.dropout_dense(hc)
		hc = F.relu(self.wordindx_decoder_dense1(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.wordindx_decoder_dense2(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.wordindx_decoder_dense3(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.wordindx_decoder_dense4(hc))
		result = self.wordindx_decoder_dense5(hc)                   # [batch, dec_o_dim]
		return result

	def duration_prediction(self, concat_feat, next_landpos):
		#predict fixation duration
		concat_feat = torch.cat((concat_feat, next_landpos), dim=1)
		concat_feat = self.dropout_dense(concat_feat)
		result = F.relu(self.dur_decoder_dense1(concat_feat))
		result = self.dropout_dense(result)
		result = F.relu(self.dur_decoder_dense2(result))
		result = self.dur_decoder_dense3(result)
		return result

	def landpos_prediction(self, hx8, sp_pos, word_enc_out, timestep):
		#predict landing position
		#retrive the word feature vector according to the target saccade word index.
		#max_sn_len is used to indicate padding, which exceeds the bounds of the maximum index of the sentence, change to SEP index for batch computation
		sp_pos_tmp = torch.where(sp_pos==self.cf["max_sn_len"], self.cf["max_sn_len"]-1, sp_pos)
		next_word_index = sp_pos_tmp[:, timestep+1]

		next_word_embed = word_enc_out[torch.arange(word_enc_out.size(0)), next_word_index]
		#mask = F.one_hot(next_word_index, word_enc_out.shape[1]).unsqueeze(2).repeat(1,1,word_enc_out.shape[2])
		#next_word_vector = torch.sum(word_enc_out * mask, 1)

		#concatenate the two hidden vectors from the two encoders
		concat_feat = torch.cat([next_word_embed, hx8],dim=1)      # [batch, units *2]

		hc = self.dropout_dense(concat_feat)
		hc = F.relu(self.landpos_decoder_dense1(hc))                  # [batch, 1]
		hc = self.dropout_dense(hc)
		hc = F.relu(self.landpos_decoder_dense2(hc))                  # [batch, 1]
		hc = self.dropout_dense(hc)
		hc = F.relu(self.landpos_decoder_dense3(hc))                  # [batch, 1]
		hc = self.dropout_dense(hc)
		hc = F.relu(self.landpos_decoder_dense4(hc))                  # [batch, 1]
		result = self.landpos_decoder_dense5(hc)                  # [batch, 1]
		return result, concat_feat

	def decode(self, sp_emd, sn_mask, sp_pos, word_enc_out, sp_fix_dur, sp_landing_pos, word_ids_sp):
		# Initialize hidden state and cell state with zeros,
		hn = torch.zeros(8, sp_emd.shape[0], self.hidden_size).to(sp_emd.device)
		hc = torch.zeros(8, sp_emd.shape[0], self.hidden_size).to(sp_emd.device)
		hx, cx = hn[0,:,:], hc[0,:,:]
		hx2, cx2 = hn[1,:,:], hc[1,:,:]
		hx3, cx3 = hn[2,:,:], hc[2,:,:]
		hx4, cx4 = hn[3,:,:], hc[3,:,:]
		hx5, cx5 = hn[4,:,:], hc[4,:,:]
		hx6, cx6 = hn[5,:,:], hc[5,:,:]
		hx7, cx7 = hn[6,:,:], hc[6,:,:]
		hx8, cx8 = hn[7,:,:], hc[7,:,:]

		dec_emb_in = self.encoder.embeddings.word_embeddings(sp_emd[:, :-1])

		if word_ids_sp is not None:
			# Pool bert subwords back to word level
			sp_merged_word_emd, sp_mask_word = self.pool_subwords_to_word(dec_emb_in,
																			word_ids_sp[:,:-1],
																			target='sp',
																			pool_method='sum')
		else:#no pooling
			sp_merged_word_emd, sp_mask_word = dec_emb_in, None


		#add positional embeddings
		position_embeddings = self.position_embeddings(sp_pos[:, :-1])
		dec_emb_in = sp_merged_word_emd+position_embeddings
		dec_emb_in = self.LayerNorm(dec_emb_in)

		dec_emb_in = dec_emb_in.permute(1,0,2)      # [step, n, emb_dim]
		dec_emb_in = self.embedding_dropout(dec_emb_in)

		#concatenate two additional gaze features
		if sp_landing_pos is not None:
			dec_emb_in = torch.cat((dec_emb_in, sp_landing_pos.permute(1,0)[:-1, :, None]), dim=2)

		if sp_fix_dur is not None:
			dec_emb_in = torch.cat((dec_emb_in, sp_fix_dur.permute(1,0)[:-1, :, None]), dim=2)

		#Predict output for each time step of the input features in turn
		output_wordindx = []
		output_dur = []
		output_landpos = []
		#atten_weights_batch = torch.empty(sp_emd.shape[0], 0, self.cf["max_sn_len"]).to(sp_emd.device)
		for i in range(dec_emb_in.shape[0]):
			hx, cx = self.decoder_cell1(dec_emb_in[i], (hx, cx))     # [batch, units]
			hx2, cx2 = self.decoder_cell2(self.dropout_LSTM(hx), (hx2, cx2))
			residual = hx2
			hx3, cx3 = self.decoder_cell3(self.dropout_LSTM(hx2), (hx3, cx3))
			input3 = hx3 + residual
			residual = input3
			hx4, cx4 = self.decoder_cell4(self.dropout_LSTM(input3), (hx4, cx4))
			input4 = hx4 + residual
			residual = input4
			hx5, cx5 = self.decoder_cell5(self.dropout_LSTM(input4), (hx5, cx5))
			input5 = hx5 + residual
			residual = input5
			hx6, cx6 = self.decoder_cell6(self.dropout_LSTM(input5), (hx6, cx6))
			input6 = hx6 + residual
			residual = input6
			hx7, cx7 = self.decoder_cell7(self.dropout_LSTM(input6), (hx7, cx7))
			input7 = hx7 + residual
			residual = input7
			hx8, cx8 = self.decoder_cell8(self.dropout_LSTM(input7), (hx8, cx8))
			input8 = hx8 + residual

			pred_loc = self.location_prediction(input8, word_enc_out, sp_pos, sn_mask, i)
			output_wordindx.append(pred_loc)

			pred_landpos, concat_feat = self.landpos_prediction(input8, sp_pos, word_enc_out, i)
			output_landpos.append(pred_landpos)

			next_landpos = sp_landing_pos[:, i+1]
			pred_dur= self.duration_prediction(concat_feat, next_landpos[:, None])
			output_dur.append(pred_dur)


		output_wordindx = torch.stack(output_wordindx,dim=0)                     # [step, batch, 1]
		output_dur = torch.stack(output_dur,dim=0)                     # [step, batch, dec_o_dim]
		output_landpos = torch.stack(output_landpos,dim=0)                     # [step, batch, dec_o_dim]
		return output_wordindx.permute(1,0,2), output_dur[:,:,0].permute(1,0), output_landpos[:,:,0].permute(1,0)     # [batch, step, dec_o_dim]


	def forward(self, sn_emd, sn_mask, sp_emd, sp_pos, word_ids_sn, word_ids_sp, sp_fix_dur, sp_landing_pos, sn_word_len):
		x, sn_mask_word = self.encode(sn_emd, sn_mask, word_ids_sn, sn_word_len)
		if sn_mask_word is None:#for Chinese dataset without token pooling
			sn_mask = torch.Tensor.bool(sn_mask)
			pred_wordindx, pred_dur, pred_landpos = self.decode(sp_emd, sn_mask, sp_pos, x, sp_fix_dur, sp_landing_pos, word_ids_sp)    # [batch, step, dec_o_dim]
		else:#for English dataset with token pooling
			pred_wordindx, pred_dur, pred_landpos = self.decode(sp_emd, sn_mask_word, sp_pos, x, sp_fix_dur, sp_landing_pos, word_ids_sp)    # [batch, step, dec_o_dim]
		return pred_wordindx, pred_dur, pred_landpos


	def scanpath_generation(self, sn_emd, sn_mask, word_ids_sn, sn_word_len, le, max_pred_len=50, device='cpu', decode_method='temp', alpha=1, k=3, prob=0.9):
		#compute the scan path generated from the model when the first CLS taken is given
		word_enc_out, sn_mask_word = self.encode(sn_emd, sn_mask, word_ids_sn, sn_word_len)
		if sn_mask_word is None:
			sn_mask = torch.Tensor.bool(sn_mask)
		else:
			sn_mask = sn_mask_word
		sn_len = torch.sum(sn_mask, axis=1)-2

		#decode
		# Initialize hidden state and cell state with zeros,
		hn = torch.zeros(8, sn_emd.shape[0], self.hidden_size).to(sn_emd.device)
		hc = torch.zeros(8, sn_emd.shape[0], self.hidden_size).to(sn_emd.device)
		hx, cx = hn[0,:,:], hc[0,:,:]
		hx2, cx2 = hn[1,:,:], hc[1,:,:]
		hx3, cx3 = hn[2,:,:], hc[2,:,:]
		hx4, cx4 = hn[3,:,:], hc[3,:,:]
		hx5, cx5 = hn[4,:,:], hc[4,:,:]
		hx6, cx6 = hn[5,:,:], hc[5,:,:]
		hx7, cx7 = hn[6,:,:], hc[6,:,:]
		hx8, cx8 = hn[7,:,:], hc[7,:,:]

		dec_in_start = (torch.ones(sn_mask.shape[0], dtype=torch.int64) * self.cf["start_token"]).to(sn_mask.device)
		dec_emb_in = self.encoder.embeddings.word_embeddings(dec_in_start) # [batch, emb_dim]

		#add positional embeddings
		start_pos = torch.zeros(sn_mask.shape[0], dtype=torch.int64).to(sn_mask.device)
		position_embeddings = self.position_embeddings(start_pos)
		dec_emb_in = dec_emb_in+position_embeddings
		dec_emb_in = self.LayerNorm(dec_emb_in)
		#concatenate two additional gaze features, which are set to zeros for CLS token
		dec_in = torch.cat((dec_emb_in, torch.zeros(dec_emb_in.shape[0],2).to(sn_emd.device)), dim=1)

		#generate fixation one by one in an autoregressive way
		output_word_index = torch.empty(sn_emd.shape[0], 0, dtype=torch.int64).to(sn_emd.device)
		output_landpos = torch.empty(sn_emd.shape[0], 0).to(sn_emd.device)
		output_dur = torch.empty(sn_emd.shape[0], 0).to(sn_emd.device)

		pred_counter = 0
		output_word_index = torch.cat([output_word_index, start_pos.unsqueeze(1)], dim=1)
		output_landpos = torch.cat([output_landpos, start_pos.unsqueeze(1)], dim=1) #zero
		output_dur = torch.cat([output_dur, start_pos.unsqueeze(1)], dim=1) #zero
		for p in range(max_pred_len):
			hx, cx = self.decoder_cell1(dec_in, (hx, cx))     # [batch, units]
			hx2, cx2 = self.decoder_cell2(self.dropout_LSTM(hx), (hx2, cx2))
			residual = hx2
			hx3, cx3 = self.decoder_cell3(self.dropout_LSTM(hx2), (hx3, cx3))
			input3 = hx3 + residual
			residual = input3
			hx4, cx4 = self.decoder_cell4(self.dropout_LSTM(input3), (hx4, cx4))
			input4 = hx4 + residual
			residual = input4
			hx5, cx5 = self.decoder_cell5(self.dropout_LSTM(input4), (hx5, cx5))
			input5 = hx5 + residual
			residual = input5
			hx6, cx6 = self.decoder_cell6(self.dropout_LSTM(input5), (hx6, cx6))
			input6 = hx6 + residual
			residual = input6
			hx7, cx7 = self.decoder_cell7(self.dropout_LSTM(input6), (hx7, cx7))
			input7 = hx7 + residual
			residual = input7
			hx8, cx8 = self.decoder_cell8(self.dropout_LSTM(input7), (hx8, cx8))
			input8 = hx8 + residual

			pred_word_idx = self.location_prediction(input8, word_enc_out, output_word_index, sn_mask, p)
			#sampling next fixation location according to the distribution
			if decode_method == 'temp':
				dist = Categorical(logits=pred_word_idx/alpha)
				#pred_idx_sampled = torch.multinomial(pred_prob, 1)
				pred_idx_sampled = dist.sample()
			elif decode_method == 'greedy':
				pred_idx_sampled = pred_word_idx.argmax(dim=-1)
			elif decode_method == 'top':
				zeros = pred_word_idx.new_ones(pred_word_idx.shape) * float('-inf')
				values, indices = torch.topk(pred_word_idx, k, dim=-1)
				zeros.scatter_(-1, indices, values)
				dist = Categorical(logits=zeros/alpha)
				pred_idx_sampled = dist.sample()
			elif decode_method == 'nucleus':
				probs=self.softmax(pred_word_idx)
				sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
				cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
				sorted_indices_to_remove = cumulative_probs > prob
				sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
				sorted_indices_to_remove[:, 0] = 0
				sorted_samp_probs = sorted_probs.clone()
				sorted_samp_probs[sorted_indices_to_remove] = 0
				sampled_sorted_indexes = sorted_samp_probs.multinomial(1).view(-1,1)
				pred_idx_sampled = sorted_indices.gather(-1, sampled_sorted_indexes).squeeze(-1)

			pred_sac_range = [le.classes_[pred_idx_sampled[i]] for i in torch.arange(pred_idx_sampled.shape[0])]
			pred_sac_range = torch.from_numpy(np.array(pred_sac_range)).to(sn_emd.device)
			pred_word_index = output_word_index[:, -1] + pred_sac_range

			#post-hoc processing: larger than sentence max length -- set to sentence length+1, i.e. token <'SEP'>
			input_ids = []
			for i in range(pred_word_index.shape[0]):
				if pred_word_index[i] > sn_len[i]:
					pred_word_index[i] = sn_len[i]+1
				elif pred_word_index[i] < 1:
					pred_word_index[i] = 1
				#compute the token ids for the predicted new tokens
				if word_ids_sn is not None:
					input_ids.append(sn_emd[i, word_ids_sn[i,:] == pred_word_index[i]])
				else:
					input_ids.append(sn_emd[i, pred_word_index[i]])
			output_word_index = torch.cat([output_word_index, pred_word_index.unsqueeze(1)], dim=1)

			pred_landpos, concat_feat = self.landpos_prediction(input8, output_word_index, word_enc_out, p)
			output_landpos = torch.cat([output_landpos, pred_landpos], dim=1)

			pred_landpos_normalzed = (pred_landpos - self.cf["landing_pos_mean"])/self.cf["landing_pos_std"]
			pred_dur= self.duration_prediction(concat_feat, pred_landpos_normalzed)
			output_dur = torch.cat([output_dur, pred_dur], dim=1)

			#prepare next timestamp input token
			pred_counter += 1
			if word_ids_sn is not None:
				#merge tokens
				dec_emb_in = torch.empty(0, self.encoder_config.hidden_size).to(sn_emd.device)
				for id in input_ids:
					dec_emb_in = torch.cat([dec_emb_in, torch.sum(self.encoder.embeddings.word_embeddings(id), axis=0)[None,:]], dim=0)

			else:
				input_ids = torch.stack(input_ids)
				dec_emb_in = self.encoder.embeddings.word_embeddings(input_ids) # [batch, emb_dim]

			#add positional embeddings
			position_embeddings = self.position_embeddings(output_word_index[:, -1])
			dec_emb_in = dec_emb_in+position_embeddings
			dec_emb_in = self.LayerNorm(dec_emb_in)
			#concatenate with gaze duration features
			pred_dur_normalzed = (pred_dur - self.cf["fix_dur_mean"])/self.cf["fix_dur_std"]
			dec_in = torch.cat((dec_emb_in, pred_landpos_normalzed), dim=1)
			dec_in = torch.cat((dec_in, pred_dur_normalzed), dim=1)
		return output_word_index,  output_landpos, output_dur                        # [batch, step]



class Eyettention2_readerID(nn.Module):
	def __init__(self, cf):
		super(Eyettention2_readerID, self).__init__()
		self.cf = cf
		self.window_width = 1
		self.hidden_size = 128
		self.atten_type = cf["atten_type"]
		self.sub_emb_size = cf["emb_size"]

		#encoder
		encoder_config = BertConfig.from_pretrained(self.cf["model_pretrained"])
		encoder_config.output_hidden_states=True
		 # initiate Bert with pre-trained weights
		print("keeping Bert with pre-trained weights")
		self.encoder = BertModel.from_pretrained(self.cf["model_pretrained"],
													config = encoder_config,
													add_pooling_layer=False)
		#freeze the parameters in Bert model
		for param in self.encoder.parameters():
			param.requires_grad = False

		self.embedding_dropout = nn.Dropout(0.4)
		self.encoder_lstm1 = nn.LSTM(input_size = 768, hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm2 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm3 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm4 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm5 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm6 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm7 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm8 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)

		#decoder
		#TODO: try relative position embedding?
		self.position_embeddings = nn.Embedding(encoder_config.max_position_embeddings, encoder_config.hidden_size)
		self.LayerNorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.layer_norm_eps)
		self.sub_embeddings = nn.Embedding(400, self.sub_emb_size)
		self.attn_position = nn.Linear(self.hidden_size, self.hidden_size+1) #acoount for the word length feature

		#initialize eight decoder cells
		self.decoder_cell1 = nn.LSTMCell(768+2+self.sub_emb_size, self.hidden_size) #acoount for fixiation duration and landing position features
		self.decoder_cell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell4 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell5 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell6 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell7 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell8 = nn.LSTMCell(self.hidden_size, self.hidden_size)

		#fixation postion decoder
		self.wordindx_decoder_dense1 = nn.Linear(self.hidden_size*2+1, 512)
		self.wordindx_decoder_dense2 = nn.Linear(512, 256)
		self.wordindx_decoder_dense3 = nn.Linear(256, 256)
		self.wordindx_decoder_dense4 = nn.Linear(256, 256)
		self.wordindx_decoder_dense5 = nn.Linear(256, self.cf["max_sn_len"]*2-3)

		#fixation duration decoder
		self.dur_decoder_dense1 = nn.Linear(self.hidden_size*2+2, 1024) # inlude the additional feature of fixation landing position
		self.dur_decoder_dense2 = nn.Linear(1024, 64)
		self.dur_decoder_dense3 = nn.Linear(64, 1)

		#landing position decoder
		self.landpos_decoder_dense1 = nn.Linear(self.hidden_size*2+1, 128)
		self.landpos_decoder_dense2 = nn.Linear(128, 128)
		self.landpos_decoder_dense3 = nn.Linear(128, 128)
		self.landpos_decoder_dense4 = nn.Linear(128, 128)
		self.landpos_decoder_dense5 = nn.Linear(128, 1)

		self.dropout_LSTM = nn.Dropout(0.2)
		self.dropout_dense = nn.Dropout(0.2)
		#for scanpath generation
		self.softmax = nn.Softmax(dim=1)


	def pool_subwords_to_word(self, subword_emb, word_ids_sn, target, pool_method='sum'):
		#try batching computing
		# Pool bert subwords back to word level
		merged_word_att = torch.empty(subword_emb.shape[0], 0, 768).to(subword_emb.device)
		if target == 'sn':
			max_len = self.cf["max_sn_len"] #CLS and SEP included
		elif target == 'sp':
			max_len = self.cf["max_sp_len"] - 1 #do not account the 'SEP' token

		for word_idx in range(max_len):
			word_mask = (word_ids_sn == word_idx).unsqueeze(2).repeat(1, 1, 768)
			#pooling method -> sum
			if pool_method=='sum':
				pooled_word_emb = torch.sum(subword_emb * word_mask, 1).unsqueeze(1) #[batch, 1, 768]
			elif pool_method=='mean':
				pooled_word_emb = torch.mean(subword_emb * word_mask, 1).unsqueeze(1) #[batch, 1, 768]
			merged_word_att = torch.cat([merged_word_att, pooled_word_emb], dim=1)
		mask_word = torch.sum(merged_word_att, 2).bool()
		return merged_word_att, mask_word


	def encode(self, sn_emd, sn_mask, word_ids_sn, sn_word_len):
		outputs = self.encoder(input_ids=sn_emd, attention_mask=sn_mask)
		hidden_rep_orig = outputs[0]
		if word_ids_sn != None:
			# Pool bert subwords back to word level for english corpus
			merged_word_att, sn_mask_word = self.pool_subwords_to_word(hidden_rep_orig,
																		word_ids_sn,
																		target='sn',
																		pool_method='sum')
		else:#no pooling for Chinese corpus
			merged_word_att, sn_mask_word = hidden_rep_orig, None

		hidden_rep = self.embedding_dropout(merged_word_att)
		#eight LSTM layers for encoder
		x, (hn, hc) = self.encoder_lstm1(hidden_rep, None)
		x, (hn, hc) = self.encoder_lstm2(self.dropout_LSTM(x), None)
		residual = x
		x, (hn, hc) = self.encoder_lstm3(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm4(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm5(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm6(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm7(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm8(self.dropout_LSTM(x), None)
		x = x + residual

		#concatenate with the word length feature
		x = torch.cat((x, sn_word_len[:, :, None]), dim=2)
		return x, sn_mask_word

	def location_prediction(self, sp_enc_out, word_enc_out, sp_pos, sn_mask, timestep):
		#predict fixation location
		# General Attention:
		# score(ht,hs) = (ht^T)(Wa)hs
		# hs is the output from encoder
		# ht is the previous hidden state from decoder
		# self.attn(o): [batch, step, units]
		attn_prod = torch.matmul(self.attn_position(sp_enc_out.unsqueeze(1)), word_enc_out.permute(0,2,1)) # [batch, 1, step]
		#local attention
		aligned_position = sp_pos[:, timestep]
		# Get window borders
		left = torch.where(aligned_position - self.window_width >= 0, (aligned_position - self.window_width), 0)
		right = torch.where(aligned_position + self.window_width <= self.cf["max_sn_len"]-1, aligned_position + self.window_width, self.cf["max_sn_len"]-1)

		#exclude padding tokens
		#only consider words in the window
		sen_seq = torch.arange(self.cf["max_sn_len"])[None,:].expand(sn_mask.shape[0],self.cf["max_sn_len"]).to(sn_mask.device)
		outside_win_mask = (sen_seq < left.unsqueeze(1)) +  (sen_seq > right.unsqueeze(1))
		attn_prod += (~sn_mask + outside_win_mask).unsqueeze(1) * -1e9
		#attn_prod += (torch.Tensor.bool(1-sn_mask_word) + outside_win_mask).unsqueeze(1) * -1e9
		att_weight = softmax(attn_prod, dim=2)             # [batch, 1, step]
		if self.atten_type == 'local-g': # local attention with Gaussian Distribution
			#gauss = lambda s: torch.exp(-torch.square(s - aligned_position.unsqueeze(1)) / (2 * torch.square(torch.tensor(self.window_width / 2))))
			gauss = lambda s: torch.exp(-torch.square(s - aligned_position.unsqueeze(1)) / (2 * torch.square(torch.tensor(self.window_width*self.cf["gaus_sd"]))))
			gauss_factor = gauss(sen_seq)
			att_weight = att_weight * gauss_factor.unsqueeze(1)

		#atten_weights_batch = torch.cat([atten_weights_batch, att_weight], dim=1)
		context = torch.matmul(att_weight, word_enc_out)    # [batch, 1, units]
		hc = torch.cat([context.squeeze(1),sp_enc_out],dim=1)      # [batch, units *2]

		hc = self.dropout_dense(hc)
		hc = F.relu(self.wordindx_decoder_dense1(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.wordindx_decoder_dense2(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.wordindx_decoder_dense3(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.wordindx_decoder_dense4(hc))
		result = self.wordindx_decoder_dense5(hc)                   # [batch, dec_o_dim]
		return result

	def duration_prediction(self, concat_feat, next_landpos):
		#predict fixation duration
		concat_feat = torch.cat((concat_feat, next_landpos), dim=1)
		concat_feat = self.dropout_dense(concat_feat)
		result = F.relu(self.dur_decoder_dense1(concat_feat))
		result = self.dropout_dense(result)
		result = F.relu(self.dur_decoder_dense2(result))
		result = self.dur_decoder_dense3(result)
		return result

	def landpos_prediction(self, hx8, sp_pos, word_enc_out, timestep):
		#predict landing position
		#retrive the word feature vector according to the target saccade word index.
		#max_sn_len is used to indicate padding, which exceeds the bounds of the maximum index of the sentence, change to SEP index for batch computation
		sp_pos_tmp = torch.where(sp_pos==self.cf["max_sn_len"], self.cf["max_sn_len"]-1, sp_pos)
		next_word_index = sp_pos_tmp[:, timestep+1]
		next_word_embed = word_enc_out[torch.arange(word_enc_out.size(0)), next_word_index]
		#mask = F.one_hot(next_word_index, word_enc_out.shape[1]).unsqueeze(2).repeat(1,1,word_enc_out.shape[2])
		#next_word_vector = torch.sum(word_enc_out * mask, 1)

		#concatenate the two hidden vectors from the two encoders
		concat_feat = torch.cat([next_word_embed, hx8],dim=1)      # [batch, units *2]

		hc = self.dropout_dense(concat_feat)
		hc = F.relu(self.landpos_decoder_dense1(hc))                  # [batch, 1]
		hc = self.dropout_dense(hc)
		hc = F.relu(self.landpos_decoder_dense2(hc))                  # [batch, 1]
		hc = self.dropout_dense(hc)
		hc = F.relu(self.landpos_decoder_dense3(hc))                  # [batch, 1]
		hc = self.dropout_dense(hc)
		hc = F.relu(self.landpos_decoder_dense4(hc))                  # [batch, 1]
		result = self.landpos_decoder_dense5(hc)                  # [batch, 1]
		return result, concat_feat

	def decode(self, sp_emd, sn_mask, sp_pos, word_enc_out, sp_fix_dur, sp_landing_pos, word_ids_sp, sub_id):
		# Initialize hidden state and cell state with zeros,
		hn = torch.zeros(8, sp_emd.shape[0], self.hidden_size).to(sp_emd.device)
		hc = torch.zeros(8, sp_emd.shape[0], self.hidden_size).to(sp_emd.device)
		hx, cx = hn[0,:,:], hc[0,:,:]
		hx2, cx2 = hn[1,:,:], hc[1,:,:]
		hx3, cx3 = hn[2,:,:], hc[2,:,:]
		hx4, cx4 = hn[3,:,:], hc[3,:,:]
		hx5, cx5 = hn[4,:,:], hc[4,:,:]
		hx6, cx6 = hn[5,:,:], hc[5,:,:]
		hx7, cx7 = hn[6,:,:], hc[6,:,:]
		hx8, cx8 = hn[7,:,:], hc[7,:,:]

		dec_emb_in = self.encoder.embeddings.word_embeddings(sp_emd[:, :-1])
		if word_ids_sp is not None:
			# Pool bert subwords back to word level
			sp_merged_word_emd, sp_mask_word = self.pool_subwords_to_word(dec_emb_in,
																			word_ids_sp[:,:-1],
																			target='sp',
																			pool_method='sum')
		else:#no pooling
			sp_merged_word_emd, sp_mask_word = dec_emb_in, None


		#add positional embeddings
		position_embeddings = self.position_embeddings(sp_pos[:, :-1])
		dec_emb_in = sp_merged_word_emd+position_embeddings
		dec_emb_in = self.LayerNorm(dec_emb_in)

		dec_emb_in = dec_emb_in.permute(1,0,2)      # [step, n, emb_dim]
		dec_emb_in = self.embedding_dropout(dec_emb_in)

		#concatenate two additional gaze features
		if sp_landing_pos is not None:
			dec_emb_in = torch.cat((dec_emb_in, sp_landing_pos.permute(1,0)[:-1, :, None]), dim=2)

		if sp_fix_dur is not None:
			dec_emb_in = torch.cat((dec_emb_in, sp_fix_dur.permute(1,0)[:-1, :, None]), dim=2)

		if sub_id is not None:
			dec_emb_in = torch.cat((dec_emb_in, self.sub_embeddings(sub_id).repeat(dec_emb_in.shape[0], 1, 1)), dim=2)

		#Predict output for each time step of the input features in turn
		output_wordindx = []
		output_dur = []
		output_landpos = []
		for i in range(dec_emb_in.shape[0]):
			hx, cx = self.decoder_cell1(dec_emb_in[i], (hx, cx))     # [batch, units]
			hx2, cx2 = self.decoder_cell2(self.dropout_LSTM(hx), (hx2, cx2))
			residual = hx2
			hx3, cx3 = self.decoder_cell3(self.dropout_LSTM(hx2), (hx3, cx3))
			input3 = hx3 + residual
			residual = input3
			hx4, cx4 = self.decoder_cell4(self.dropout_LSTM(input3), (hx4, cx4))
			input4 = hx4 + residual
			residual = input4
			hx5, cx5 = self.decoder_cell5(self.dropout_LSTM(input4), (hx5, cx5))
			input5 = hx5 + residual
			residual = input5
			hx6, cx6 = self.decoder_cell6(self.dropout_LSTM(input5), (hx6, cx6))
			input6 = hx6 + residual
			residual = input6
			hx7, cx7 = self.decoder_cell7(self.dropout_LSTM(input6), (hx7, cx7))
			input7 = hx7 + residual
			residual = input7
			hx8, cx8 = self.decoder_cell8(self.dropout_LSTM(input7), (hx8, cx8))
			input8 = hx8 + residual

			pred_loc = self.location_prediction(input8, word_enc_out, sp_pos, sn_mask, i)
			output_wordindx.append(pred_loc)

			pred_landpos, concat_feat = self.landpos_prediction(input8, sp_pos, word_enc_out, i)
			output_landpos.append(pred_landpos)

			next_landpos = sp_landing_pos[:, i+1]
			pred_dur= self.duration_prediction(concat_feat, next_landpos[:, None])
			output_dur.append(pred_dur)


		output_wordindx = torch.stack(output_wordindx,dim=0)                     # [step, batch, 1]
		output_dur = torch.stack(output_dur,dim=0)                     # [step, batch, dec_o_dim]
		output_landpos = torch.stack(output_landpos,dim=0)                     # [step, batch, dec_o_dim]
		return output_wordindx.permute(1,0,2), output_dur.permute(1,0,2), output_landpos.permute(1,0,2)     # [batch, step, dec_o_dim]


	def forward(self, sn_emd, sn_mask, sp_emd, sp_pos, word_ids_sn, word_ids_sp, sp_fix_dur, sp_landing_pos, sn_word_len, sub_id):
		x, sn_mask_word = self.encode(sn_emd, sn_mask, word_ids_sn, sn_word_len)
		if sn_mask_word is None:#for Chinese dataset without token pooling
			sn_mask = torch.Tensor.bool(sn_mask)
			pred_wordindx, pred_dur, pred_landpos = self.decode(sp_emd, sn_mask, sp_pos, x, sp_fix_dur, sp_landing_pos, word_ids_sp, sub_id)    # [batch, step, dec_o_dim]
		else:#for English dataset with token pooling
			pred_wordindx, pred_dur, pred_landpos = self.decode(sp_emd, sn_mask_word, sp_pos, x, sp_fix_dur, sp_landing_pos, word_ids_sp, sub_id)    # [batch, step, dec_o_dim]
		return pred_wordindx, pred_dur, pred_landpos
