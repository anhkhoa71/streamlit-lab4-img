import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchPositionEmbedding(nn.Module):
    def __init__(self, img_size=224, in_channels=3, patch_size=16, emb_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=emb_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embs = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

    def forward(self, x):
        x = self.conv2d(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        batch_size = x.shape[0]
        cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)
        pos_embs = self.pos_embs.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x + pos_embs

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, emb_dim=768, num_heads=12, attn_dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)
        x, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False)
        return x

class MLPBlock(nn.Module):
    def __init__(self, emb_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        return self.mlp(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim=768, num_heads=12, attn_dropout=0.0, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.msa_block = MultiheadSelfAttentionBlock(emb_dim, num_heads, attn_dropout)
        self.mlp_block = MLPBlock(emb_dim, mlp_size, dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, in_channels=3, patch_size=16, emb_dim=768, emb_dropout=0.1,
                 num_transformer_layers=12, num_heads=12, attn_dropout=0.0, mlp_size=3072, dropout=0.1,
                 num_classes=6):
        super().__init__()
        self.pat_pos_emb = PatchPositionEmbedding(img_size, in_channels, patch_size, emb_dim)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.encoder = nn.Sequential(
            *[TransformerEncoderBlock(emb_dim, num_heads, attn_dropout, mlp_size, dropout)
              for _ in range(num_transformer_layers)]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        x = self.pat_pos_emb(x)
        x = self.emb_dropout(x)
        x = self.encoder(x)
        x = x[:, 0, :]
        return self.mlp_head(x)

class MixupCutmix:
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob

    def _rand_bbox(self, W, H, lam, device):
        cut_rat = torch.sqrt(1. - lam)
        cut_w = (W * cut_rat).int()
        cut_h = (H * cut_rat).int()

        cx = torch.randint(0, W, (1,), device=device)
        cy = torch.randint(0, H, (1,), device=device)

        x1 = torch.clamp(cx - cut_w // 2, 0, W)
        x2 = torch.clamp(cx + cut_w // 2, 0, W)
        y1 = torch.clamp(cy - cut_h // 2, 0, H)
        y2 = torch.clamp(cy + cut_h // 2, 0, H)

        return x1.item(), y1.item(), x2.item(), y2.item()

    def __call__(self, images, labels):
        labels = F.one_hot(torch.tensor(labels), num_classes=6).float()
        if torch.rand(1).item() > self.prob:
            return images, labels

        device = images.device
        batch_size = images.size(0)

        indices = torch.randperm(batch_size, device=device)

        x1 = images
        x2 = images[indices]
        y1 = labels
        y2 = labels[indices]

        # ---------------- MIXUP ----------------
        if torch.rand(1).item() < self.switch_prob:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().to(device)
            images = lam * x1 + (1 - lam) * x2

        # ---------------- CUTMIX ----------------
        else:
            lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().to(device)

            W = images.size(3)
            H = images.size(2)
            x1b, y1b, x2b, y2b = self._rand_bbox(W, H, lam, device)

            images[:, :, y1b:y2b, x1b:x2b] = x2[:, :, y1b:y2b, x1b:x2b]

            lam = 1 - ((x2b - x1b) * (y2b - y1b) / (W * H))

        labels = lam * y1 + (1 - lam) * y2
        return images, labels
