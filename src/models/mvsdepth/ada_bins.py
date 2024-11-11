"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from vit-pytorch (https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py;) 
# MIT license
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, 
            q # query
            ):
        n, c, h, w = x.size()
        scale = c ** -0.5
        #_, cout, cq = q.size()
        x = rearrange(x, 'b c h w -> b c (h w)')
        y = einsum('b c s1, b s2 c -> b s2 s1', x, q) # s1 = h*w
        y = rearrange(y, 'b c (h w) -> b c h w', h=h, w=w)
        y = y*scale
        return y


class ImagePatchTransformer(nn.Module):
    """Patch-based Transformer Encoder for image data.

    This module divides the input image into patches and processes them using a Transformer encoder.
    
    Attributes:
        transformer_block: A Transformer encoder that processes embedded patches.
        patch_embedding_layer: A convolutional layer to convert image patches into embeddings.
        positional_encodings: Learnable positional encodings added to patch embeddings.
    """

    def __init__(self, 
                 num_input_channels: int, 
                 patch_dimension: int = 10, 
                 embedding_size: int = 128, 
                 attention_heads: int = 4) -> None:
        """Initializes the ImagePatchTransformer.

        Args:
            num_input_channels: Number of input channels in the image (e.g., 3 for RGB).
            patch_dimension: Size of each patch (both height and width).
            embedding_size: Dimension of the patch embeddings.
            attention_heads: Number of attention heads in the Transformer encoder.
        """
        super(ImagePatchTransformer, self).__init__()
        
        encoder = nn.TransformerEncoderLayer(
            embedding_size, attention_heads, dim_feedforward=1024
        )
        
        # Takes input shape (S, N, E) with S for seq length, 
        # N for batch size, E for embedding dimension;
        self.transformer_block = nn.TransformerEncoder(
            encoder_layer = encoder, 
            num_layers=4
        )  

        self.patch_embedding_layer = nn.Conv2d(
            num_input_channels, embedding_size,
            kernel_size = patch_dimension, 
            stride = patch_dimension, 
            padding = 0
        )

        self.posi_encodings = nn.Parameter(
            torch.rand(500, embedding_size), requires_grad=True
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ImagePatchTransformer.

        Args:
            input_tensor: Input tensor of shape (batch, C, H, W).

        Returns:
            Tensor of shape (sequence_length, batch_size, embedding_size) 
            after passing through the Transformer encoder.
        """
        # [N, E, H, W] --> [N, E, S]
        patch_embed = self.patch_embedding_layer(input_tensor).flatten(start_dim=2)
        patch_embed = patch_embed + self.posi_encodings[:patch_embed.shape[2], :].T.unsqueeze(0)

        # change from [N,E,S] to [S, N, E] format which is required by Transformer
        patch_embed = patch_embed.permute(2, 0, 1)
        # size [S, N, E] after transformer 
        output = self.transformer_block(patch_embed)  
        
        return output



class mViT(nn.Module):
    def __init__(self, in_channels, 
                patch_size=16, 
                dim_out = 64,
                group_num = 1,
                embedding_dim=128, 
                num_heads=4, 
                norm='linear'
                ):
        super(mViT, self).__init__()
        self.norm = norm
        self.group_num = group_num
        self.patch_transformer = ImagePatchTransformer(
                                    num_input_channels = in_channels, 
                                    patch_dimension = patch_size, 
                                    embedding_size = embedding_dim, 
                                    attention_heads = num_heads)

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, 
                                 kernel_size=3, stride=1, padding=1)
        ndepth = dim_out
        self.regressor = nn.Sequential(
                    nn.Linear(self.group_num*embedding_dim, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, ndepth)
                )

    def normalize_bins(self, x):
        if self.norm == 'linear':
            y = torch.relu(x)
            # the small positive ensures each bin-width is strictly positive;
            eps = 1e-3 
            y = y + eps
            y = y / y.sum(dim=1, keepdim=True)
        
        elif self.norm == 'softmax':
            y = torch.softmax(x, dim=1)
        
        else:
            y = torch.sigmoid(x)
            y = y / y.sum(dim=1, keepdim=True)
        return y

    def forward(self, x, is_split=False):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x)  # .shape = S, N, E
        assert tgt.size(0) >= self.group_num, \
            f'cannot be divied to {self.group_num} groups'
        # avg_pooling to get fixed dimension;
        y = rearrange(
            tgt, '(g s) n e -> n e g s', g=self.group_num) # g=4
        y = reduce(y, 'n e g s -> n e g', 'mean')
        y = rearrange(y, 'n e g -> n (e g)')
        y = self.regressor(y)  # .shape = N, dim_out
        if is_split:
            # split to halves;
            y_left, y_right = y.chunk(2, dim=1)
            y_left  = self.normalize_bins(y_left)
            y_right = self.normalize_bins(y_right)
            y = torch.cat([y_left, y_right], dim=1)
        else:
            y = self.normalize_bins(y)
            
        bins_widths = y
        return bins_widths


"""
This code is adopted from vit-pytorch (https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py;) 
MIT license
"""

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


