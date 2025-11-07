import math
import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import ASPP


# ============================================================
# 🔹 AnyUp-based decoder (frozen, differentiable)
# ============================================================
class AnyUpDecoder(nn.Module):
    """
    Uses frozen AnyUp to upsample each window's [B, D, h, w] features into [B, D, 256, 256].
    Then compresses each to D/2 with 1x1 conv, concatenates (spatial), and predicts logits.
    """
    def __init__(self, dim=1024, num_classes=3, rgb_bands=(0, 1, 2)):
        super().__init__()
        import torch.hub
        print("📦 Using AnyUp decoder (frozen weights, gradients flow through)")
        self.anyup = torch.hub.load("wimmerth/anyup", "anyup", trust_repo=True)
        for p in self.anyup.parameters():
            p.requires_grad_(False)  # frozen but graph-enabled

        # Per-branch channel compression: 1024 -> 512
        compressed = max(dim // 2, 128)
        self.compress_A = nn.Conv2d(dim, compressed, 1)
        self.compress_B = nn.Conv2d(dim, compressed, 1)

        # Lightweight post head: (512+512)=1024 -> 512 -> C
        self.post_head = nn.Sequential(
            nn.Conv2d(2 * compressed, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, 1),
        )

        self.bnA = nn.BatchNorm2d(compressed)
        self.bnB = nn.BatchNorm2d(compressed)
        self.act = nn.ReLU(inplace=True)
        self.rgb_bands = rgb_bands

    def _norm_rgb(self, x):
        rgb = x[:, self.rgb_bands, :, :].clamp(0, 3000.0) / 3000.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (rgb - mean) / std

    def forward(self, feats_2L_D, hr_image, q_chunk_size=256):
        B, N2, D = feats_2L_D.shape
        L = N2 // 2
        h = w = int(math.sqrt(L))
        assert h * w == L, "Token length must be square (L = h*w)."

        # tokens -> 2D feature maps
        A = feats_2L_D[:, :L, :].transpose(1, 2).reshape(B, D, h, w)
        Bm = feats_2L_D[:, L:, :].transpose(1, 2).reshape(B, D, h, w)

        # split HR image for A/B (assumes stacking)
        C = hr_image.shape[1]
        mid = C // 2 
        hrA, hrB = hr_image[:, :mid, :, :], hr_image[:, mid:, :, :]

        # normalize HR RGB subsets
        hrA = self._norm_rgb(hrA)
        hrB = self._norm_rgb(hrB)

        # Frozen AnyUp (no no_grad; keep graph)
        A_up = self.anyup(hrA, A, q_chunk_size=q_chunk_size)
        B_up = self.anyup(hrB, Bm, q_chunk_size=q_chunk_size)

        # Per-branch compression
        A_up = self.act(self.bnA(self.compress_A(A_up)))
        B_up = self.act(self.bnB(self.compress_B(B_up)))

        fused = torch.cat([A_up, B_up], dim=1)  # [B, D', 256, 256]
        return self.post_head(fused)            # [B, C, 256, 256]


# ============================================================
# 🔹 Fusion modules (token space)
# ============================================================
class ConcatFusion(nn.Module):
    def forward(self, x_2L_D):
        B, N2, D = x_2L_D.shape
        L = N2 // 2
        return torch.cat([x_2L_D[:, :L, :], x_2L_D[:, L:, :]], dim=-1)


class MLPCombiner(nn.Module):
    def __init__(self, D=1024, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 2 * D
        self.mlp = nn.Sequential(
            nn.Linear(2 * D, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, D)
        )
    def forward(self, x_2L_D):
        B, N2, D = x_2L_D.shape
        L = N2 // 2
        AB = torch.cat([x_2L_D[:, :L, :], x_2L_D[:, L:, :]], dim=-1)
        return self.mlp(AB)


class BiCrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.0):
        super().__init__()
        self.normA = nn.LayerNorm(embed_dim)
        self.normB = nn.LayerNorm(embed_dim)
        self.crossA = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.crossB = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj  = nn.Linear(2 * embed_dim, embed_dim)
        self.norm  = nn.LayerNorm(embed_dim)
    def forward(self, x_2L_D):
        B, N2, D = x_2L_D.shape
        L = N2 // 2
        A = x_2L_D[:, :L, :]
        Bm = x_2L_D[:, L:, :]
        A_, _ = self.crossA(self.normA(A), self.normB(Bm), self.normB(Bm))
        B_, _ = self.crossB(self.normB(Bm), self.normA(A), self.normA(A))
        fused = torch.cat([A + A_, Bm + B_], dim=-1)
        return self.norm(self.proj(fused))


# ============================================================
# 🔹 Standard decoders (token → 256×256)
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim)
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(x + self.block(x))


class ConvDecoderHead(nn.Module):
    def __init__(self, dim, patch_size, num_classes):
        super().__init__()
        hidden, C_out, r = 512, 64, patch_size
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, C_out * r * r, 3, padding=1),
            nn.PixelShuffle(r),
            nn.Conv2d(C_out, num_classes, kernel_size=1)
        )
    def forward(self, x_L_D):
        B, L, D = x_L_D.shape
        h = w = int(math.sqrt(L))
        x = x_L_D.transpose(1, 2).reshape(B, D, h, w)
        return self.decoder(x)


class ViTDecoderHead(nn.Module):
    def __init__(self, dim, patch_size, num_classes):
        super().__init__()
        hidden, C_out, r = 512, 64, patch_size
        self.proj_in = nn.Conv2d(dim, hidden, 3, padding=1)
        self.res1 = ResidualBlock(hidden)
        self.res2 = ResidualBlock(hidden)
        self.aspp = ASPP(in_channels=hidden, atrous_rates=(6, 12, 18), out_channels=hidden)
        self.conv_ps = nn.Conv2d(hidden, C_out * r * r, 3, padding=1)
        self.shuffle = nn.PixelShuffle(r)
        self.out_conv = nn.Conv2d(C_out, num_classes, kernel_size=1)
    def forward(self, x_L_D):
        B, L, D = x_L_D.shape
        h = w = int(math.sqrt(L))
        x = x_L_D.transpose(1, 2).reshape(B, D, h, w)
        x = self.proj_in(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.aspp(x)
        x = self.shuffle(self.conv_ps(x))
        return self.out_conv(x)


class SegmentationHead(nn.Module):
    """
    fusion_type: 'concat' | 'mlp' | 'bca'
    decoder_type: 'conv' | 'vit' | 'anyup'
    """
    def __init__(self, fusion_type="concat", decoder_type="vit",
                 dim=1024, patch_size=16, num_classes=3, bca_heads=8, rgb_bands=(0,1,2)):
        super().__init__()
        assert fusion_type in ["concat", "mlp", "bca"]
        assert decoder_type in ["conv", "vit", "anyup"]

        self.decoder_type = decoder_type
        self.dim = dim
        self.patch_size = patch_size

        # Fusion
        if decoder_type == "anyup":
            self.fuse = None
        else:
            if fusion_type == "concat":
                self.fuse = ConcatFusion()
                self.fused_dim = 2 * dim
            elif fusion_type == "mlp":
                self.fuse = MLPCombiner(D=dim)
                self.fused_dim = dim
            else:
                self.fuse = BiCrossAttentionFusion(embed_dim=dim, num_heads=bca_heads)
                self.fused_dim = dim

        # Decoder
        if decoder_type == "conv":
            self.decoder = ConvDecoderHead(self.fused_dim, patch_size, num_classes)
        elif decoder_type == "vit":
            self.decoder = ViTDecoderHead(self.fused_dim, patch_size, num_classes)
        else:
            self.decoder = AnyUpDecoder(dim=dim, num_classes=num_classes, rgb_bands=rgb_bands)

    def forward(self, x):
        # import code;code.interact(local=dict(globals(), **locals()));
        if isinstance(x, dict):
            feats = x["feats"]

            # 🔄 Backward compatibility: handle old (B, 2, N, D) format
            if feats.ndim == 4 and feats.shape[1] == 2:
                feats = feats.reshape(feats.shape[0], -1, feats.shape[-1])
            else:
                AssertionError("Input 'feat' must be of shape (B, 2, N, D) or (B, N2, D).")
            if self.decoder_type == "anyup":
                hr_image = x.get("image", None)
                if hr_image is None:
                    raise ValueError("decoder_type='anyup' requires 'image' in input dict.")
                return self.decoder(feats, hr_image)
            fused_tokens = self.fuse(feats)
            return self.decoder(fused_tokens)
        
        else:
            feats = x
            fused_tokens = self.fuse(feats)
            return self.decoder(fused_tokens)
        

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda:3")
    B, N, D, C = 2, 1024, 768, 3     # ✅ 8×8 patches
    patch_size = 8
    feats = torch.randn(B, 2, N, D, device=device)
    hr = torch.randn(B, 8, 256, 256, device=device)

    fusions = ["concat", "mlp", "bca"]
    decoders = ["anyup", "conv", "vit", "anyup"]

    for dec in decoders:
        fusion_list = ["concat"] if dec == "anyup" else fusions
        for fus in fusion_list:
            print(f"\n🔹 Testing fusion={fus:12s} | decoder={dec}")
            model = SegmentationHead(
                fusion_type=fus,
                decoder_type=dec,
                dim=D,
                patch_size=patch_size,
                num_classes=C,
                bca_heads=8,
                rgb_bands=(0,1,2)
            ).to(device)
            out = model({"feat": feats, "image": hr})
            print("   ✅ Output:", tuple(out.shape))