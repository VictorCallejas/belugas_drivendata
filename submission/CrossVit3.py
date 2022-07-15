import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# pre-layernorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feedforward
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

# attention
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer encoder, for qall and large patches
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# projecting CLS tokens, in the case that qall and large patch tokens have different dimensions
class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# cross attention transformer
class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(ProjectInOut(dim, dim, PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))))

    def forward(self, q_tokens, ref_tokens):
        (q_cls, q_patch_tokens), (ref_cls, ref_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (q_tokens, ref_tokens))
        for attend in self.layers:
            q_cls = attend(q_cls, context = ref_patch_tokens, kv_include_self = True) + q_cls
            ref_cls = attend(ref_cls, context = q_patch_tokens, kv_include_self = True) + ref_cls
            
        q_tokens = torch.cat((q_cls, q_patch_tokens), dim = 1)
        ref_tokens = torch.cat((ref_cls, ref_patch_tokens), dim = 1)
        
        return q_tokens, ref_tokens

# multi-scale encoder
class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        enc_depth,
        dim,
        enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for enc_d, cross_d in zip(enc_depth, cross_attn_depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = dim, dropout = dropout, depth = enc_d, **enc_params),
                CrossTransformer(dim = dim, depth = cross_d, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, q_tokens, ref_tokens):
        for enc, cross_attend in self.layers:
            q_tokens, ref_tokens = enc(q_tokens), enc(ref_tokens)
            q_tokens, ref_tokens = cross_attend(q_tokens, ref_tokens)

        return q_tokens, ref_tokens

# patch-based image to token embedder
class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)

# cross ViT class
class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size=224,
        num_classes=1,
        dim=192,
        patch_size = 16,
        enc_depth = [2,1,1],
        enc_heads = 8,
        enc_mlp_dim = 2048,
        enc_dim_head = 64,
        cross_attn_depth = [1,1,2],
        cross_attn_heads = 12,
        cross_attn_dim_head = 64,
        dropout = 0.2,
        emb_dropout = 0.1
    ):
        super().__init__()
        self.image_embedder = ImageEmbedder(dim = dim, image_size = image_size, patch_size = patch_size, dropout = emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            dim = dim,
            enc_depth = enc_depth,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            enc_params = dict(
                heads = enc_heads,
                mlp_dim = enc_mlp_dim,
                dim_head = enc_dim_head
            ),
            dropout = dropout
        )

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, query, reference):
        
        q_tokens = self.image_embedder(query)
        ref_tokens = self.image_embedder(reference)

        q_tokens, ref_tokens = self.multi_scale_encoder(q_tokens, ref_tokens)

        q_cls, ref_cls = map(lambda t: t[:, 0], (q_tokens, ref_tokens))

        #cls = torch.cat([q_cls, ref_cls], dim=1)
        cls = q_cls + ref_cls
        logits = self.mlp_head(cls)

        return logits
    

def crossvit_base_224():
    
    return CrossViT(
        image_size=224,
        num_classes=1,
        dim=192,
        patch_size = 32,
        enc_depth = [2,2,1],
        enc_heads = 6,
        enc_mlp_dim = 512,
        enc_dim_head = 32,
        cross_attn_depth = [1,1,2],
        cross_attn_heads = 12,
        cross_attn_dim_head = 64,
        dropout = 0.2,
        emb_dropout = 0.1
        )