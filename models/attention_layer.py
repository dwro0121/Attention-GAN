import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import sys
from .mask import RTLMask_layer1, RTLMask_layer2, LTRMask_layer1, LTRMask_layer2


def Conv1x1(in_c, out_c):
    return nn.Conv2d(in_c, out_c, 1)


class Self_Attention(nn.Module):
    def __init__(self, in_c):
        super(Self_Attention, self).__init__()
        self.in_c = in_c

        self.query = Conv1x1(self.in_c, self.in_c // 8)
        self.key = Conv1x1(self.in_c, self.in_c // 8)
        self.value = Conv1x1(self.in_c, self.in_c)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        query = self.query(x).view(B, C // 8, W * H).permute(0, 2, 1)
        key = self.key(x).view(B, C // 8, W * H)
        value = self.value(x).view(B, C, W * H)
        attention_map = F.softmax(torch.bmm(query, key), -1)
        out = torch.bmm(value, attention_map.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        out = self.gamma * out + x

        return out


class Criss_Cross_Attention(nn.Module):
    def __init__(self, in_c):
        super(Criss_Cross_Attention, self).__init__()
        self.in_c = in_c

        self.query = Conv1x1(self.in_c, self.in_c // 8)
        self.key = Conv1x1(self.in_c, self.in_c // 8)
        self.value = Conv1x1(self.in_c, self.in_c)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        query_h = query.permute(0, 3, 2, 1).contiguous().view(B * W, H, C // 8)
        query_w = query.permute(0, 2, 3, 1).contiguous().view(B * H, W, C // 8)
        key_h = key.permute(0, 3, 1, 2).contiguous().view(B * W, C // 8, H)
        key_w = key.permute(0, 2, 1, 3).contiguous().view(B * H, C // 8, W)
        value_h = value.permute(0, 3, 1, 2).contiguous().view(B * W, C, H)
        value_w = value.permute(0, 2, 1, 3).contiguous().view(B * H, C, W)

        energy_h = (
            torch.bmm(query_h, key_h).contiguous().view(B, W, H, H)
            - (torch.eye(W, W).unsqueeze(0).repeat(B * W, 1, 1) * sys.maxsize)
            .contiguous()
            .view(B, W, H, H)
            .cuda()
        ).permute(0, 2, 1, 3)
        energy_w = torch.bmm(query_w, key_w).contiguous().view(B, H, W, W)

        attention_map = F.softmax(torch.cat([energy_h, energy_w], 3), 3)
        attention_map_h = (
            attention_map[:, :, :, 0:H]
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B * W, H, H)
            .permute(0, 2, 1)
        )
        attention_map_w = (
            attention_map[:, :, :, H : H + W]
            .contiguous()
            .view(B * H, W, W)
            .permute(0, 2, 1)
        )
        out_h = (
            torch.bmm(value_h, attention_map_h)
            .contiguous()
            .view(B, W, -1, H)
            .permute(0, 2, 3, 1)
        )
        out_w = (
            torch.bmm(value_w, attention_map_w)
            .contiguous()
            .view(B, H, -1, W)
            .permute(0, 2, 1, 3)
        )
        out = out_h + out_w
        out = out.view(B, C, W, H)
        out = self.gamma * out + x

        return out


class CCA(nn.Module):
    def __init__(self, in_c):
        super(CCA, self).__init__()
        self.Attention = Criss_Cross_Attention(in_c)

    def forward(self, x):
        out = self.Attention(x)
        out = self.Attention(out)
        return out


class Your_Local_Attention(nn.Module):
    def __init__(self, in_c):
        super(Your_Local_Attention, self).__init__()
        self.in_c = in_c

        self.query = Conv1x1(self.in_c, self.in_c // 8)
        self.key = Conv1x1(self.in_c, self.in_c // 8)
        self.value = Conv1x1(self.in_c, self.in_c // 2)
        self.after_attention = Conv1x1(self.in_c // 2, self.in_c)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.mask_rtl1 = RTLMask_layer1
        self.mask_rtl2 = RTLMask_layer2
        self.mask_ltr1 = LTRMask_layer1
        self.mask_ltr2 = LTRMask_layer2

    def forward(self, x):
        B, C, W, H = x.size()

        head_num = 8
        head_size = C // (8 * head_num)

        query = self.query(x).view(-1, head_num, head_size, H * W)
        key = F.max_pool2d(self.key(x), kernel_size=2, stride=2, padding=0).view(
            -1, head_num, head_size, H * W // 4
        )
        attention_logits = torch.einsum("abcd, abce -> abde", query, key)

        masks = self.get_grid_masks((H, W), (H // 2, W // 2))
        attention_adder = (1.0 - masks) * (-1000.0)
        attention_adder = torch.from_numpy(attention_adder).cuda()

        attention_logits += attention_adder
        attention_map = F.softmax(attention_logits, dim=-1)

        value = F.max_pool2d(self.value(x), kernel_size=2, stride=2, padding=0)
        value_head_size = C // (2 * head_num)
        value = value.view(-1, head_num, value_head_size, H * W // 4)

        out = torch.einsum("abcd, abed -> abec", attention_map, value)
        out = out.contiguous().view(-1, C // 2, W, H)
        out = self.after_attention(out)
        out = self.gamma * out + x

        return out

    def get_grid_masks(self, gridO, gridI):
        masks = []

        # RTL
        masks.append(self.mask_rtl1.get_mask(gridI, nO=gridO))
        masks.append(self.mask_rtl2.get_mask(gridI, nO=gridO))

        masks.append(self.mask_rtl1.get_mask(gridI, nO=gridO))
        masks.append(self.mask_rtl2.get_mask(gridI, nO=gridO))

        # LTR
        masks.append(self.mask_ltr1.get_mask(gridI, nO=gridO))
        masks.append(self.mask_ltr2.get_mask(gridI, nO=gridO))

        masks.append(self.mask_ltr1.get_mask(gridI, nO=gridO))
        masks.append(self.mask_ltr2.get_mask(gridI, nO=gridO))

        return np.array(masks)


def Attention_Layer(in_c, attention="SA"):
    if attention == "SA":
        return Self_Attention(in_c)
    elif attention == "CCA":
        return CCA(in_c)
    elif attention == "YLA":
        return Your_Local_Attention(in_c)
    else:
        return None
