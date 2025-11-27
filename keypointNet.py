import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def spatial_softmax(logits, tau=1.0):
    """Apply a spatial softmax over each heatmap in the tensor."""
    batch, channels, height, width = logits.shape
    if tau < 1e-8:
        tau = 1e-8
    scaled = logits / tau
    scaled = scaled - scaled.amax(dim=(2, 3), keepdim=True)
    prob_flat = torch.softmax(scaled.flatten(2), dim=-1)
    return prob_flat.view(batch, channels, height, width)


def soft_argmax_2d(logits, tau=1.0, return_prob=False):
    """Return expected (x, y) coordinates and optionally the spatial probabilities."""
    batch, _, height, width = logits.shape
    prob = spatial_softmax(logits, tau=tau)
    xs = torch.arange(width, device=logits.device, dtype=logits.dtype).view(1, 1, 1, width)
    ys = torch.arange(height, device=logits.device, dtype=logits.dtype).view(1, 1, height, 1)
    mu_x = (prob * xs).sum(dim=(2, 3))
    mu_y = (prob * ys).sum(dim=(2, 3))
    coords = torch.stack([mu_x, mu_y], dim=-1)
    if return_prob:
        return coords, prob
    return coords

def combined_heatmap_coord_loss(
    logits,
    gt_xy,
    gt_heatmaps,
    tau=1.0,
    delta=1.0,
    valid_mask=None,
    lambda_coord=1.0,
    use_kl_for_heatmap=False,
):
    """Composite heatmap + coordinate loss used during training."""
    pred_xy, prob = soft_argmax_2d(logits, tau=tau, return_prob=True)
    # heatmap loss
    if use_kl_for_heatmap:
        heatmap_loss = F.kl_div(prob.log(), gt_heatmaps, reduction='batchmean')
    else:
        heatmap_loss = F.mse_loss(prob, gt_heatmaps)
    # coordinate loss
    per_coord = F.smooth_l1_loss(pred_xy, gt_xy, reduction='none', beta=delta)
    per_kpt = per_coord.sum(dim=-1)
    px_err = torch.linalg.norm(pred_xy - gt_xy, dim=-1)
    if valid_mask is not None:
        keep = valid_mask > 0
        per_kpt = per_kpt[keep]
        px_err = px_err[keep]
    coord_loss = per_kpt.mean()
    mean_px_err = px_err.mean()
    total_loss = heatmap_loss + lambda_coord * coord_loss
    return total_loss, heatmap_loss, coord_loss, mean_px_err, pred_xy, prob


class SelfAttention(nn.Module):
    """Lightweight multi-head self-attention block for 2D feature maps."""

    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        if in_channels % num_heads != 0:
            raise ValueError("in_channels must be divisible by num_heads")
        self.h = num_heads
        self.d = in_channels // num_heads
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        q = self.q(x).view(batch, self.h, self.d, height * width).transpose(2, 3)
        k = self.k(x).view(batch, self.h, self.d, height * width)
        v = self.v(x).view(batch, self.h, self.d, height * width).transpose(2, 3)
        attn = torch.matmul(q, k) / (self.d ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(batch, channels, height, width)
        return self.proj(out) + x

class ASPP(nn.Module):
    """Atrous spatial pyramid pooling block."""

    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, 1)
        self.c2 = nn.Conv2d(in_channels, out_channels, 3, padding=3, dilation=3)
        self.c3 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.c4 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
        )
        self.out = nn.Conv2d(out_channels * 5, out_channels, 1)

    def forward(self, x):
        height, width = x.shape[-2:]
        feats = [
            self.c1(x),
            self.c2(x),
            self.c3(x),
            self.c4(x),
            F.interpolate(self.pool(x), size=(height, width), mode="bilinear", align_corners=False),
        ]
        return self.out(torch.cat(feats, dim=1))

class KeypointDetectionNet(nn.Module):
    """Joint keypoint heatmap and mask prediction network."""

    def __init__(self, num_keypoints=11, use_bn=True, dropout_rate=0.1, seg_channels=128):
        super().__init__()
        self.K = num_keypoints
        self.with_mask = True

        self.adapter_rgb = nn.Conv2d(5, 3, 1)

        resnet = models.resnet34(weights=None)
        self.enc_conv1 = resnet.conv1
        self.enc_bn1 = resnet.bn1
        self.enc_relu = resnet.relu
        self.enc_pool = resnet.maxpool
        self.l1, self.l2, self.l3, self.l4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        self.attn_bottleneck = SelfAttention(512)
        self.aspp = ASPP(512, 256)
        self.dropout = nn.Dropout(p=dropout_rate)
        norm_layer = (lambda channels: nn.BatchNorm2d(channels)) if use_bn else (
            lambda channels: nn.GroupNorm(32, channels)
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(256, seg_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            norm_layer(seg_channels),
            nn.Dropout(p=dropout_rate / 2),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
            nn.Conv2d(seg_channels, seg_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(seg_channels // 2, 1, 1)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        self.rgb_proj = nn.Conv2d(256, 256, 1)
        self.mask_proj = nn.Conv2d(1, 256, 1)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c1  = nn.Conv2d(256 + 256, 256, 3, padding=1); self.n1 = norm_layer(256)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c2  = nn.Conv2d(256 + 128, 128, 3, padding=1); self.n2 = norm_layer(128)
        self.attn_dec = SelfAttention(128)  # keep ONE decoder attention
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c3  = nn.Conv2d(128 + 64, 64, 3, padding=1);   self.n3 = norm_layer(64)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c4  = nn.Conv2d(64 + 64, 32, 3, padding=1);    self.n4 = norm_layer(32)
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c5  = nn.Conv2d(32, 32, 3, padding=1);         self.n5 = norm_layer(32)
        self.out = nn.Conv2d(32, self.K, 3, padding=1)


    def cat_skip(self, x, skip):
        """Safely concatenate skip features by trimming spatial mismatches."""
        _, _, hx, wx = x.shape
        _, _, hs, ws = skip.shape
        height = min(hx, hs)
        width = min(wx, ws)
        if hx != height or wx != width:
            x = x[..., :height, :width]
        if hs != height or ws != width:
            skip = skip[..., :height, :width]
        return torch.cat([x, skip], dim=1)


    def coord_channels(self,B,H,W,device,dtype):
        ys = torch.linspace(-1, 1, H, dtype=dtype, device=device).view(H, 1).expand(H, W)
        xs = torch.linspace(-1, 1, W, dtype=dtype, device=device).view(1, W).expand(H, W)
        xx = xs.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        yy = ys.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        return xx, yy

    def assemble_rgb_only(self, img, view_type):
        """Concatenate RGB (or grayscale) with coordinate channels and adapt to backbone."""
        batch, _, height, width = img.shape
        channels = [img] if view_type == "rgb" else [img.mean(1, keepdim=True)]
        xx, yy = self.coord_channels(batch, height, width, img.device, img.dtype)
        channels.extend([xx, yy])
        stacked = torch.cat(channels, dim=1)
        return self.adapter_rgb(stacked)

    def forward(self, img_crop, view_type="rgb"):
        """Forward pass returning heatmap logits and segmentation logits."""
        batch, channels, height, width = img_crop.shape
        assert channels == 3 and view_type == "rgb"

        x = self.assemble_rgb_only(img_crop, view_type)
        x = self.enc_conv1(x)
        x = self.enc_bn1(x)
        x = self.enc_relu(x)
        s0 = x
        x = self.enc_pool(x)
        x = self.l1(x)
        s1 = x
        x = self.l2(x)
        s2 = x
        x = self.l3(x)
        s3 = x
        x = self.l4(x)
        x = self.attn_bottleneck(x)
        bottleneck = self.aspp(x)
        x = self.dropout(bottleneck)

        pred_mask_logit = self.seg_head(bottleneck)
        pred_mask_prob = torch.sigmoid(pred_mask_logit)

        mask_resized = F.interpolate(pred_mask_prob, size=bottleneck.shape[-2:], mode='bilinear')
        mask_flat = self.mask_proj(mask_resized).flatten(2).transpose(1, 2)
        rgb_flat = self.rgb_proj(bottleneck).flatten(2).transpose(1, 2)
        fused_flat, _ = self.cross_attn(query=rgb_flat, key=rgb_flat, value=mask_flat)
        fused = fused_flat.transpose(1, 2).view(batch, 256, *bottleneck.shape[-2:])
        enhanced_bottleneck = fused + bottleneck

        x = self.up1(enhanced_bottleneck)
        x = self.cat_skip(x, s3)
        x = self.n1(F.relu(self.c1(x), inplace=True))
        x = self.up2(x)
        x = self.cat_skip(x, s2)
        x = self.n2(F.relu(self.c2(x), inplace=True))
        x = self.attn_dec(x)
        x = self.up3(x)
        x = self.cat_skip(x, s1)
        x = self.n3(F.relu(self.c3(x), inplace=True))
        x = self.up4(x)
        x = self.cat_skip(x, s0)
        x = self.n4(F.relu(self.c4(x), inplace=True))
        x = self.up5(x)
        x = self.n5(F.relu(self.c5(x), inplace=True))
        logits = self.out(x)

        return logits, pred_mask_logit
    

def gaussian_heatmap(
    height,
    width,
    center_xy,
    sigma,
    device,
    normalize="sum",
):
    """Generate a normalized 2D Gaussian heatmap centered at a pixel coordinate."""
    x_coords = torch.arange(0, width, dtype=torch.float32, device=device)
    y_coords = torch.arange(0, height, dtype=torch.float32, device=device)
    x_grid = x_coords.view(1, width).expand(height, width)
    y_grid = y_coords.view(height, 1).expand(height, width)
    cx, cy = center_xy
    dist2 = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    denom = 2.0 * (sigma ** 2)
    heatmap = torch.exp(-dist2 / denom)
    if normalize == "max":
        max_value = heatmap.max()
        heatmap = heatmap / max_value if max_value > 0 else heatmap
    elif normalize == "sum":
        total = heatmap.sum()
        heatmap = heatmap / total if total > 0 else heatmap
    return heatmap


def generate_gt_heatmaps(
    gt_points_xy,
    height,
    width,
    sigma,
    device,
    normalize="sum",
):
    """Create Gaussian supervision heatmaps for each keypoint."""
    batch, kpts, _ = gt_points_xy.shape
    heatmaps = torch.zeros(batch, kpts, height, width, device=device, dtype=torch.float32)
    for b in range(batch):
        for k in range(kpts):
            x = gt_points_xy[b, k, 0]
            y = gt_points_xy[b, k, 1]
            if (x >= 0) and (x < width) and (y >= 0) and (y < height):
                heatmaps[b, k] = gaussian_heatmap(height, width, (x, y), sigma, device, normalize)
    return heatmaps