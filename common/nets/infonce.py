import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import numpy as np
__all__ = ['InfoNCE', 'info_nce']
from einops import rearrange, reduce, repeat

class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = dist.get_rank()
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )

def sample_feature_from_3d(feature_space_3d, joint_img):
    """
    Args:
        feature_space_3d: (B, C, D, H, W)
        joint_img: (B, J, 3)
    """
    
def intra_info_nce_loss(feature_space_3d, gt_joint_img, pred_joint_img, joint_mask, anchour_num=50, pos_num=100, neg_num=200, temperature=0.1):
    """
    Args:
        feature_space_3d: (B, C, D, H, W)
        gt_joint_img: (B, J, 3)
        pred_joint_img: (B, J, 3)
        joint_mask: (B, J)
    """
    B, J, _ = gt_joint_img.shape
    anchour_idx = np.random.choice(B * J, anchour_num, replace=B * J < anchour_num) # (A, )
    anchour_pred_joint = pred_joint_img.reshape(-1, 3)[anchour_idx] # (A, 3)
    anchour_batch_idx = anchour_idx // J # (A, )
    anchour_label = anchour_idx % J # (A, )
    # sample the same label joint form gt for positive set
    loss_valid = []
    positive_key = []
    nagative_key = []
    sample_gt_joint_feat = F.grid_sample(feature_space_3d, gt_joint_img.unsqueeze(1).unsqueeze(1), mode='bilinear', align_corners=True).squeeze(2).squeeze(2).permute(0, 2, 1).contiguous() # B, J, C
    sample_pred_joint_feat = F.grid_sample(feature_space_3d, pred_joint_img.unsqueeze(1).unsqueeze(1), mode='bilinear', align_corners=True).squeeze(2).squeeze(2).permute(0, 2, 1).contiguous() # B, J, C
    for label in anchour_label:
        label_mask = joint_mask[:, label] == 1
        if torch.sum(label_mask) == 0: # 没有正样本
            loss_valid.append(False)
            pos_idx = np.random.choice(B, pos_num, replace=True)
        else:
            loss_valid.append(True)
            # sample positive set
            pos_idx = np.random.choice(np.where(label_mask.cpu().numpy())[0], pos_num, replace=torch.sum(label_mask).item() < pos_num)
        pos_gt_joint = sample_gt_joint_feat[pos_idx, label] # (P, C)
        positive_key.append(pos_gt_joint)

        # sample negative set (pred的其他joint)
        neg_label_mask = np.arange(B * J) % J != label
        neg_idx = np.random.choice(np.where(neg_label_mask)[0], neg_num, replace=np.sum(neg_label_mask) < neg_num)
        batch_idx = neg_idx // J
        joint_idx = neg_idx % J
        neg_pred_joint = sample_pred_joint_feat[batch_idx, joint_idx] # (N, C)
        nagative_key.append(neg_pred_joint)
    anchour_faet = sample_pred_joint_feat[anchour_batch_idx, anchour_label] # (A, C)
    positive_key = torch.cat(positive_key, dim=0) # (A * P, C)
    nagative_key = torch.stack(nagative_key, dim=0) # (A, N, C)
    loss_valid = torch.tensor(loss_valid, dtype=torch.bool, device=anchour_faet.device) # (A, )

    # loss
    anchour_faet = repeat(anchour_faet, 'a c -> (a p) c', p=pos_num) # (A, P, C)
    nagative_key = repeat(nagative_key, 'a n c -> (a p) n c', p=pos_num) # (A * P, N, C)
    loss = info_nce(anchour_faet, positive_key, negative_keys=nagative_key, temperature=temperature, reduction='none', negative_mode='paired')
    loss = loss.view(anchour_num, pos_num) # (A, P)
    loss = loss[loss_valid].mean()
    return loss

def inter_info_nce_loss(feature_space_3d, pred_joint_img, anchour_num=50, pos_num=100, neg_num=200, temperature=0.1):
    """
    Args:
        feature_space_3d: (B, C, D, H, W)
        pred_joint_img: (B, J, 3)
    """
    B, J, _ = pred_joint_img.shape
    anchour_idx = np.random.choice(B * J, anchour_num, replace=B * J < anchour_num) # (A, )
    anchour_batch_idx = anchour_idx // J # (A, )
    anchour_label = anchour_idx % J # (A, )
    # sample the same label joint form gt for positive set
    loss_valid = []
    positive_key = []
    nagative_key = []
    sample_pred_joint_feat = F.grid_sample(feature_space_3d, pred_joint_img.unsqueeze(1).unsqueeze(1), mode='bilinear', align_corners=True).squeeze(2).squeeze(2).permute(0, 2, 1).contiguous() # B, J, C
    pred_joint_img = pred_joint_img.add(1).mul(4) # (0 ~ 8) (B, J, 3) 
    up_coord = torch.ceil(pred_joint_img).long() # (B, J, 3)
    down_coord = torch.floor(pred_joint_img).long() # (B, J, 3)
    up_coord = torch.clamp(up_coord, 0, 7)
    down_coord = torch.clamp(down_coord, 0, 7)
    positive_mask = torch.ones(B, 8, 8, 8, device=feature_space_3d.device).bool()
    feature_space_3d = rearrange(feature_space_3d, 'b c d h w -> (b d h w) c').contiguous() # (B * D * H * W, C)
    for b_id in range(B):
        cur_mask = positive_mask[b_id]
        cur_up_coord = up_coord[b_id]
        cur_down_coord = down_coord[b_id]
        cur_mask[cur_up_coord[:, 0], cur_up_coord[:, 1], cur_up_coord[:, 2]] = False
        cur_mask[cur_down_coord[:, 0], cur_up_coord[:, 1], cur_up_coord[:, 2]] = False
        cur_mask[cur_up_coord[:, 0], cur_down_coord[:, 1], cur_up_coord[:, 2]] = False
        cur_mask[cur_up_coord[:, 0], cur_up_coord[:, 1], cur_down_coord[:, 2]] = False
        cur_mask[cur_down_coord[:, 0], cur_down_coord[:, 1], cur_up_coord[:, 2]] = False
        cur_mask[cur_down_coord[:, 0], cur_up_coord[:, 1], cur_down_coord[:, 2]] = False
        cur_mask[cur_up_coord[:, 0], cur_down_coord[:, 1], cur_down_coord[:, 2]] = False
        cur_mask[cur_down_coord[:, 0], cur_down_coord[:, 1], cur_down_coord[:, 2]] = False
        positive_mask[b_id] = cur_mask
    neg_idx_list = np.where(positive_mask.flatten().cpu().numpy())[0]

    for a_idx, b_id, label in zip(anchour_idx, anchour_batch_idx, anchour_label):
        # sample positive set
        pos_idx_pool = np.where(np.arange(B * J) != a_idx)[0]
        pos_idx = np.random.choice(pos_idx_pool, pos_num, replace=len(pos_idx_pool) < pos_num)
        pos_batch_idx = pos_idx // J
        pos_label = pos_idx % J
        pos_pred_joint = sample_pred_joint_feat[pos_batch_idx, pos_label] # (P, C)
        positive_key.append(pos_pred_joint)

        # sample negative set （pred joint 之外的 voxel）
    neg_idx = np.random.choice(neg_idx_list, neg_num, replace=len(neg_idx_list) < neg_num)
    
    neg_feat = feature_space_3d[neg_idx] # (N, C)
    anchour_feat = sample_pred_joint_feat[anchour_batch_idx, anchour_label] # (A, C)
    anchour_feat = repeat(anchour_feat, 'a c -> (a p) c', p=pos_num) # (A * P, C)
    pos_faet = torch.cat(positive_key, dim=0) # (A * P, C)
    loss = info_nce(anchour_feat, pos_faet, negative_keys=neg_feat, temperature=temperature, reduction='mean', negative_mode='unpaired')
    # loss = loss.view(anchour_num, pos_num) # (A, P)
    return loss




if __name__ == '__main__':
    loss = InfoNCE()
    batch_size, embedding_size = 32, 128
    query = torch.randn(batch_size, embedding_size)
    positive_key = torch.randn(batch_size, embedding_size)
    output = loss(query, positive_key)
    print(output)