"""RADVL model fusion."""
import torch
from torch import nn

from radvl.model_components import RADVL as OriginalRADVL, dynamic_covariance_regularizer, seed_torch


class RADVLFusion(nn.Module):
    """Eqs. (32)-(42): bidirectional contexts, difference gates and alignment."""
    def __init__(self,dimension,tau):
        super().__init__()
        self.tau=tau
        self.scale=dimension**-0.5
        self.query=nn.Linear(dimension,dimension,bias=False)
        self.key=nn.Linear(dimension,dimension,bias=False)
        self.value=nn.Linear(dimension,dimension,bias=False)
        self.gate_1=nn.Linear(3*dimension,dimension)
        self.gate_2=nn.Linear(3*dimension,dimension)

    def context(self,first,second):
        attention=torch.softmax((self.query(first)@self.key(second).T)*self.scale,dim=-1)
        return attention@self.value(second)

    def forward(self,similarity,association):
        context_1=self.context(similarity,association)
        context_2=self.context(association,similarity)
        delta_1=similarity-context_1
        delta_2=association-context_2
        gate_1=torch.sigmoid(self.gate_1(torch.cat([delta_1,similarity,context_1],dim=-1)))
        gate_2=torch.sigmoid(self.gate_2(torch.cat([delta_2,association,context_2],dim=-1)))
        first=gate_1*similarity+(1-gate_1)*context_1
        second=gate_2*association+(1-gate_2)*context_2
        discrepancy=delta_1.square().sum(dim=-1)+delta_2.square().sum(dim=-1)
        weight=torch.exp(-discrepancy/self.tau)
        alignment=(weight*(first-second).square().sum(dim=-1)).mean()
        fused=first+second
        with torch.no_grad():
            self.diagnostics=dict(delta_squared_mean=discrepancy.mean().item(),
                alignment_strength_mean=weight.mean().item(),alignment_strength_min=weight.min().item(),
                alignment_strength_max=weight.max().item(),
                alignment_weight_below_1e_minus8_fraction=(weight<1e-8).float().mean().item(),
                gate_1_mean=gate_1.mean().item(),gate_2_mean=gate_2.mean().item(),
                fused_norm_mean=fused.norm(dim=1).mean().item())
            self.diagnostic_arrays=dict(delta_1=delta_1.detach(),delta_2=delta_2.detach(),
                gate_1=gate_1.detach(),gate_2=gate_2.detach(),alignment_strength=weight.detach())
        return fused,alignment


class RADVL(OriginalRADVL):
    """Similarity concat -> cross-view fusion -> elementwise pair product -> MLP.

    Deliberately scoped: original association+similarity CoVAE encoder input,
    posterior-mean decoding, dynamic cross-covariance and KL scaling remain.
    This is not a full rewrite of the manuscript association-view objective.
    """
    def __init__(self,mi_num,dis_num,hidd_list,num_proj_hidden,hyperpm,
                 num_in_node=218,num_in_edge=271,num_hidden1=512,num_out=64,num_layers=15):
        super().__init__(mi_num,dis_num,hidd_list,num_proj_hidden,hyperpm,
                         num_in_node,num_in_edge,num_hidden1,num_out,num_layers)
        dimension=hyperpm.hid_dim
        self.dr_similarity_projection=nn.Linear(2*dimension,dimension)
        self.circ_similarity_projection=nn.Linear(2*dimension,dimension)
        self.dr_association_projection=nn.Identity() if num_out==dimension else nn.Linear(num_out,dimension)
        self.circ_association_projection=nn.Identity() if num_out==dimension else nn.Linear(num_out,dimension)
        # Replace the old GIN/HGCN-only differential-attention fusion modules.
        self.dr_fusion=RADVLFusion(dimension,hyperpm.fusion_tau)
        self.circ_fusion=RADVLFusion(dimension,hyperpm.fusion_tau)
        self.pair_predictor=nn.Sequential(nn.Linear(dimension,hyperpm.pair_hidden_dim),
                                         nn.ReLU(),nn.Linear(hyperpm.pair_hidden_dim,1))
        self.pair_chunk_rows=hyperpm.pair_chunk_rows

    def pair_logits(self,circ,drug):
        chunks=[]
        for start in range(0,len(circ),self.pair_chunk_rows):
            pairs=circ[start:start+self.pair_chunk_rows,None,:]*drug[None,:,:]
            chunks.append(self.pair_predictor(pairs).squeeze(-1))
        return torch.cat(chunks,dim=0)

    def forward(self,concat_circ_tensor,concat_dr_tensor,G_circ_Kn,G_circ_Km,
                G_dr_Kn,G_dr_Km,AT,A,dr_sim_graph,circ_sim_graph):
        device=concat_circ_tensor.device
        dr_sim_graph=dr_sim_graph.to(device)
        circ_sim_graph=circ_sim_graph.to(device)
        dr_global=self.linear_x(self.hgcn_dr(self.dr_fastkan(concat_dr_tensor),G_dr_Kn))
        circ_global=self.linear_y(self.hgcn_circ(self.circ_fastkan(concat_circ_tensor),G_circ_Kn))
        dr_local=self.dr_gin_encoder(dr_sim_graph.x,dr_sim_graph.edge_index)
        circ_local=self.circ_gin_encoder(circ_sim_graph.x,circ_sim_graph.edge_index)
        dr_similarity=self.dr_similarity_projection(torch.cat([dr_local,dr_global],dim=-1))
        circ_similarity=self.circ_similarity_projection(torch.cat([circ_local,circ_global],dim=-1))
        circ_encoded=self.node_encoders1(concat_circ_tensor)
        drug_encoded=self.hyperedge_encoders1(concat_dr_tensor)
        (self.z_node_mean,self.z_node_log_std,self.z_node_sample,
         self.z_edge_mean,self.z_edge_log_std,self.z_edge_sample)=self.sample_latent(circ_encoded,drug_encoded)
        reconstruction=self.decoder2(self.z_node_mean,self.z_edge_mean)
        circ_association=self.circ_association_projection(self.z_node_mean.float())
        dr_association=self.dr_association_projection(self.z_edge_mean.float())
        circ_fused,circ_alignment=self.circ_fusion(circ_similarity,circ_association)
        dr_fused,drug_alignment=self.dr_fusion(dr_similarity,dr_association)
        alignment=circ_alignment+drug_alignment
        self.prediction_logits=self.pair_logits(circ_fused,dr_fused)
        score=torch.sigmoid(self.prediction_logits)
        covariance=dynamic_covariance_regularizer(self.z_node_mean,reconstruction,tau=self.co_tau)
        self.diagnostic_scores=dict(
            association=torch.sigmoid(self.z_node_mean@self.z_edge_mean.T).float().detach(),
            fused_inner_product=torch.sigmoid(circ_fused@dr_fused.T).detach(),
            combined=score.detach())
        self.diagnostic_embeddings=dict(drug=dr_fused.detach(),circ=circ_fused.detach())
        return score,reconstruction,dr_fused,circ_fused,covariance,alignment
