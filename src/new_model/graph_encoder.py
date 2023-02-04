from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn
import numpy as np

def truncated_normal(t, mean=0.0, std=0.01):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
      cond = torch.logical_or(t < mean - 2*std, t > mean + 2*std)
      if not torch.sum(cond):
        break
      t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
    return t


class multi_head_attention(torch.nn.Module):
    def __init__(self,d_key,d_value,d_model,n_head,attention_dropout, device='cuda', L_config="L_none"):
        super(multi_head_attention,self).__init__()
        self.d_key=d_key
        self.d_value=d_value
        self.d_model=d_model
        self.n_head=n_head
        self.attention_dropout=attention_dropout
        self._device = device
        self._L_config = L_config

        if self._L_config == "L_none" or self._L_config == "L_edge":
            self.layer_q=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_q.weight.data=truncated_normal(self.layer_q.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_q.bias, 0.0)
            self.layer_k=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_k.weight.data=truncated_normal(self.layer_k.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_k.bias, 0.0)
            self.layer_v=torch.nn.Linear(self.d_model,self.d_value * self.n_head)
            self.layer_v.weight.data=truncated_normal(self.layer_v.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_v.bias, 0.0)

        elif self._L_config == "L_node" or self._L_config == "L_node_edge":
            self.layer_qs=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_qs.weight.data=truncated_normal(self.layer_qs.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_qs.bias, 0.0)
            self.layer_qr=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_qr.weight.data=truncated_normal(self.layer_qr.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_qr.bias, 0.0)
            self.layer_qo=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_qo.weight.data=truncated_normal(self.layer_qo.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_qo.bias, 0.0)
            self.layer_qa=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_qa.weight.data=truncated_normal(self.layer_qa.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_qa.bias, 0.0)
            self.layer_qv=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_qv.weight.data=truncated_normal(self.layer_qv.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_qv.bias, 0.0)


            self.layer_ks=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_ks.weight.data=truncated_normal(self.layer_ks.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_ks.bias, 0.0)
            self.layer_kr=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_kr.weight.data=truncated_normal(self.layer_kr.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_kr.bias, 0.0)
            self.layer_ko=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_ko.weight.data=truncated_normal(self.layer_ko.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_ko.bias, 0.0)
            self.layer_ka=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_ka.weight.data=truncated_normal(self.layer_ka.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_ka.bias, 0.0)
            self.layer_kv=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_kv.weight.data=truncated_normal(self.layer_kv.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_kv.bias, 0.0)

            self.layer_vs=torch.nn.Linear(self.d_model,self.d_value * self.n_head)
            self.layer_vs.weight.data=truncated_normal(self.layer_vs.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_vs.bias, 0.0)
            self.layer_vr=torch.nn.Linear(self.d_model,self.d_value * self.n_head)
            self.layer_vr.weight.data=truncated_normal(self.layer_vr.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_vr.bias, 0.0)
            self.layer_vo=torch.nn.Linear(self.d_model,self.d_value * self.n_head)
            self.layer_vo.weight.data=truncated_normal(self.layer_vo.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_vo.bias, 0.0)
            self.layer_va=torch.nn.Linear(self.d_model,self.d_value * self.n_head)
            self.layer_va.weight.data=truncated_normal(self.layer_va.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_va.bias, 0.0)
            self.layer_vv=torch.nn.Linear(self.d_model,self.d_value * self.n_head)
            self.layer_vv.weight.data=truncated_normal(self.layer_vv.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_vv.bias, 0.0)

        else:
            self.layer_qe=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_qe.weight.data=truncated_normal(self.layer_qe.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_qe.bias, 0.0)
            self.layer_qr=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_qr.weight.data=truncated_normal(self.layer_qr.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_qr.bias, 0.0)            
            self.layer_ke=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_ke.weight.data=truncated_normal(self.layer_ke.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_ke.bias, 0.0)
            self.layer_kr=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
            self.layer_kr.weight.data=truncated_normal(self.layer_kr.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_kr.bias, 0.0)
            self.layer_ve=torch.nn.Linear(self.d_model,self.d_value * self.n_head)
            self.layer_ve.weight.data=truncated_normal(self.layer_ve.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_ve.bias, 0.0)
            self.layer_vr=torch.nn.Linear(self.d_model,self.d_value * self.n_head)
            self.layer_vr.weight.data=truncated_normal(self.layer_vr.weight.data,std=0.02)
            torch.nn.init.constant_(self.layer_vr.bias, 0.0)

        self.project_layer=torch.nn.Linear(d_value * n_head,self.d_model)
        self.project_layer.weight.data=truncated_normal(self.project_layer.weight.data,std=0.02)
        torch.nn.init.constant_(self.project_layer.bias, 0.0)

    def forward(self,
                queries,
                edges_query,
                edges_key,
                edges_value,
                attn_bias):
        #B is batch_size, M is max_seq_len, N is n_head, H is d_key
        batch_size=queries.size(0)
        max_seq_len=queries.size(1)
        #query,key,value is [B,M,N*H], edges_key,edges_value is [M,M,H], attn_bias is [B,N,M,M]
        keys = queries 
        values = keys 
        #q,k,v is [B,N,M,H]

        # mask_s=torch.tensor([1]+[0]*(max_seq_len-1)).to('cuda')
        # mask_r=torch.tensor([0,1]+[0]*(max_seq_len-2)).to('cuda')
        # mask_o=torch.tensor([0,0,1]+[0]*(max_seq_len-3)).to('cuda')
        # mask_a=torch.tensor([0,0,0]+[1,0]*int(((max_seq_len-3)/2))).to('cuda')
        # mask_v=torch.tensor([0,0,0]+[0,1]*int(((max_seq_len-3)/2))).to('cuda')

        mask_s=torch.tensor([1]+[0]*(max_seq_len-1)).to(self._device)
        mask_r=torch.tensor([0,1]+[0]*(max_seq_len-2)).to(self._device)
        mask_o=torch.tensor([0,0,1]+[0]*(max_seq_len-3)).to(self._device)
        mask_a=torch.tensor([0,0,0]+[1,0]*int(((max_seq_len-3)/2))).to(self._device)
        mask_v=torch.tensor([0,0,0]+[0,1]*int(((max_seq_len-3)/2))).to(self._device)

        if self._L_config == "L_none" or self._L_config == "L_edge":
            q=self.layer_q(queries) 
            q=q.view(batch_size,-1,self.n_head,self.d_key).transpose(1,2)   
            k=self.layer_k(keys) 
            k=k.view(batch_size,-1,self.n_head,self.d_key).transpose(1,2) 
            v=self.layer_v(values) 
            v=v.view(batch_size,-1,self.n_head,self.d_key).transpose(1,2) 

        elif self._L_config == "L_node" or self._L_config == "L_node_edge":
            queries_s=torch.mul(queries,mask_s[:,None].expand(-1,queries.size(-1)))
            queries_r=torch.mul(queries,mask_r[:,None].expand(-1,queries.size(-1)))
            queries_o=torch.mul(queries,mask_o[:,None].expand(-1,queries.size(-1)))
            queries_a=torch.mul(queries,mask_a[:,None].expand(-1,queries.size(-1)))
            queries_v=torch.mul(queries,mask_v[:,None].expand(-1,queries.size(-1)))

            q_s=self.layer_qs(queries_s) 
            q_r=self.layer_qr(queries_r)
            q_o=self.layer_qo(queries_o)
            q_a=self.layer_qa(queries_a)
            q_v=self.layer_qv(queries_v)

            q=(q_s+q_r+q_o+q_a+q_v).view(batch_size,-1,self.n_head,self.d_key).transpose(1,2) 

            keys_s=torch.mul(keys,mask_s[:,None].expand(-1,keys.size(-1)))
            keys_r=torch.mul(keys,mask_r[:,None].expand(-1,keys.size(-1)))
            keys_o=torch.mul(keys,mask_o[:,None].expand(-1,keys.size(-1)))
            keys_a=torch.mul(keys,mask_a[:,None].expand(-1,keys.size(-1)))
            keys_v=torch.mul(keys,mask_v[:,None].expand(-1,keys.size(-1)))

            k_s=self.layer_ks(keys_s) 
            k_r=self.layer_kr(keys_r)
            k_o=self.layer_ko(keys_o)
            k_a=self.layer_ka(keys_a)
            k_v=self.layer_kv(keys_v)

            k=(k_s+k_r+k_o+k_a+k_v).view(batch_size,-1,self.n_head,self.d_key).transpose(1,2) 

            values_s=torch.mul(values,mask_s[:,None].expand(-1,values.size(-1)))
            values_r=torch.mul(values,mask_r[:,None].expand(-1,values.size(-1)))
            values_o=torch.mul(values,mask_o[:,None].expand(-1,values.size(-1)))
            values_a=torch.mul(values,mask_a[:,None].expand(-1,values.size(-1)))
            values_v=torch.mul(values,mask_v[:,None].expand(-1,values.size(-1)))

            v_s=self.layer_vs(values_s) 
            v_r=self.layer_vr(values_r)
            v_o=self.layer_vo(values_o)
            v_a=self.layer_va(values_a)
            v_v=self.layer_vv(values_v)

            v=(v_s+v_r+v_o+v_a+v_v).view(batch_size,-1,self.n_head,self.d_key).transpose(1,2) 

        else:
            queries_s=torch.mul(queries,mask_s[:,None].expand(-1,queries.size(-1)))
            queries_r=torch.mul(queries,mask_r[:,None].expand(-1,queries.size(-1)))
            queries_o=torch.mul(queries,mask_o[:,None].expand(-1,queries.size(-1)))
            queries_a=torch.mul(queries,mask_a[:,None].expand(-1,queries.size(-1)))
            queries_v=torch.mul(queries,mask_v[:,None].expand(-1,queries.size(-1)))

            q_s=self.layer_qe(queries_s) 
            q_r=self.layer_qr(queries_r)
            q_o=self.layer_qe(queries_o)
            q_a=self.layer_qr(queries_a)
            q_v=self.layer_qe(queries_v)

            q=(q_s+q_r+q_o+q_a+q_v).view(batch_size,-1,self.n_head,self.d_key).transpose(1,2) 

            keys_s=torch.mul(keys,mask_s[:,None].expand(-1,keys.size(-1)))
            keys_r=torch.mul(keys,mask_r[:,None].expand(-1,keys.size(-1)))
            keys_o=torch.mul(keys,mask_o[:,None].expand(-1,keys.size(-1)))
            keys_a=torch.mul(keys,mask_a[:,None].expand(-1,keys.size(-1)))
            keys_v=torch.mul(keys,mask_v[:,None].expand(-1,keys.size(-1)))

            k_s=self.layer_ke(keys_s) 
            k_r=self.layer_kr(keys_r)
            k_o=self.layer_ke(keys_o)
            k_a=self.layer_kr(keys_a)
            k_v=self.layer_ke(keys_v)

            k=(k_s+k_r+k_o+k_a+k_v).view(batch_size,-1,self.n_head,self.d_key).transpose(1,2) 

            values_s=torch.mul(values,mask_s[:,None].expand(-1,values.size(-1)))
            values_r=torch.mul(values,mask_r[:,None].expand(-1,values.size(-1)))
            values_o=torch.mul(values,mask_o[:,None].expand(-1,values.size(-1)))
            values_a=torch.mul(values,mask_a[:,None].expand(-1,values.size(-1)))
            values_v=torch.mul(values,mask_v[:,None].expand(-1,values.size(-1)))

            v_s=self.layer_ve(values_s) 
            v_r=self.layer_vr(values_r)
            v_o=self.layer_ve(values_o)
            v_a=self.layer_vr(values_a)
            v_v=self.layer_ve(values_v)

            v=(v_s+v_r+v_o+v_a+v_v).view(batch_size,-1,self.n_head,self.d_key).transpose(1,2) 

        if self._L_config == "L_none" or self._L_config == "L_node":
            scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_key)
            scores = torch.add(scores, attn_bias)
        else:
            #scores1,scores2,scores is [B,N,M,M]
            scores1 = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_key)
            scores2 = torch.matmul(q.permute(2,0,1,3).contiguous().view(max_seq_len,-1,self.d_key),edges_key.transpose(-1,-2)).view(max_seq_len,-1,self.n_head,max_seq_len).permute(1,2,0,3)/ np.sqrt(self.d_key)
            scores3 = torch.matmul(k.permute(2,0,1,3).contiguous().view(max_seq_len,-1,self.d_key),edges_query.transpose(-1,-2)).view(max_seq_len,-1,self.n_head,max_seq_len).permute(1,2,0,3)/ np.sqrt(self.d_key)
            scores4 = torch.matmul(torch.unsqueeze(edges_key,-1).transpose(-1,-2),torch.unsqueeze(edges_query,-1)).squeeze()/ np.sqrt(self.d_key)
            #scores2 = (edges_key+edges_value)
            scores=scores1+scores2+scores3+scores4
            scores=torch.add(scores,attn_bias)
            #weights is [B,N,M,M]
        weights=torch.nn.Dropout(self.attention_dropout)(torch.nn.Softmax(dim=-1)(scores))
        if self._L_config == "L_none" or self._L_config == "L_node":
            context = torch.matmul(weights, v)
        else:
            #context1,context2,context is [B,N,M,H]
            context1= torch.matmul(weights,v)
            context2= torch.matmul(weights.permute(2,0,1,3).contiguous().view(max_seq_len,-1,max_seq_len),edges_value).view(max_seq_len,-1,self.n_head,self.d_value).permute(1,2,0,3)
            context=torch.add(context1,context2)
        #output is [B,M,N*H]
        output=context.transpose(1,2).contiguous().view(batch_size,-1,self.n_head*self.d_value)
        output=self.project_layer(output)
        return output


class positionwise_feed_forward(torch.nn.Module):
    def __init__(self,d_inner_hid,d_model):
        super(positionwise_feed_forward,self).__init__()
        self.d_inner_hid=d_inner_hid
        self.d_hid=d_model

        self.fc1=torch.nn.Linear(self.d_hid,self.d_inner_hid)
        self.fc1.weight.data=truncated_normal(self.fc1.weight.data,std=0.02)
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        self.fc2=torch.nn.Linear(self.d_inner_hid,self.d_hid)
        self.fc2.weight.data=truncated_normal(self.fc2.weight.data,std=0.02)
        torch.nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self,x):       
        return self.fc2(torch.nn.GELU()(self.fc1(x)))

class encoder_layer(torch.nn.Module):
    def __init__(self,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            device,
            L_config):
        super(encoder_layer,self).__init__()
        self.n_head=n_head
        self.d_key=d_key
        self.d_value=d_value
        self.d_model=d_model
        self.d_inner_hid=d_inner_hid
        self.prepostprocess_dropout=prepostprocess_dropout
        self.attention_dropout=attention_dropout
        self._device = device
        self._L_config = L_config

        self.multi_head_attention=multi_head_attention(
            self.d_key,
            self.d_value,
            self.d_model,
            self.n_head,
            self.attention_dropout,
            self._device,
            self._L_config)
        self.layer_norm1=torch.nn.LayerNorm(normalized_shape=self.d_model,eps=1e-12,elementwise_affine=True)

        self.positionwise_feed_forward=positionwise_feed_forward(
            self.d_inner_hid,
            self.d_model)
        self.layer_norm2=torch.nn.LayerNorm(normalized_shape=self.d_model,eps=1e-12,elementwise_affine=True)

    def forward(self,enc_input,
                    edges_query,
                    edges_key,
                    edges_value,
                    attn_bias):
        attn_output = self.multi_head_attention(
            enc_input,
            edges_query,
            edges_key,
            edges_value,
            attn_bias)
        attn_output=self.layer_norm1(torch.add(enc_input,torch.nn.Dropout(self.prepostprocess_dropout)(attn_output)))
        
        ffd_output = self.positionwise_feed_forward(attn_output)
        ffd_output=self.layer_norm2(torch.add(attn_output,torch.nn.Dropout(self.prepostprocess_dropout)(ffd_output)))
        return ffd_output


class encoder(torch.nn.Module):
    def __init__(self,n_layer,n_head,d_key,d_value,d_model,
                d_inner_hid,prepostprocess_dropout,attention_dropout, device, L_config):
        super(encoder,self).__init__()
        self.n_layer=n_layer
        self.n_head=n_head
        self.d_key=d_key
        self.d_value=d_value
        self.d_model=d_model
        self.d_inner_hid=d_inner_hid
        self.prepostprocess_dropout=prepostprocess_dropout
        self.attention_dropout=attention_dropout
        self._device = device
        self._L_config = L_config

        for nl in range(self.n_layer):        
            setattr(self,"encoder_layer{}".format(nl),encoder_layer(
                self.n_head,
                self.d_key,
                self.d_value,
                self.d_model,
                self.d_inner_hid,
                self.prepostprocess_dropout,
                self.attention_dropout,
                self._device,
                self._L_config))

    def forward(self,enc_input,edges_query,edges_key,edges_value,attn_bias):
        for nl in range(self.n_layer):
            enc_output = getattr(self,"encoder_layer{}".format(nl))(
                enc_input,
                edges_query,
                edges_key,
                edges_value,
                attn_bias)
            enc_input = enc_output
        return enc_output
