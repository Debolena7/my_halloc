from typing import List

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, ResNetModel, VisualBertModel

from src.model import register_model


@register_model(name="halloc")
class HallocModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        vlm_out_dim: int = 4096,
    ) -> None:
        super().__init__()

        hidden_dim = 768
        visual_embedding_dim = 512

        self.img_seq_len = 1

        self.proj = nn.Linear(vlm_out_dim, hidden_dim)

        # Vision encoder (ResNet-50)
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.vision_encoder = ResNetModel.from_pretrained("microsoft/resnet-50")

        # Project CNN features → 512 (VisualBERT visual dim)
        self.vision_proj = nn.Linear(
            2048 * 7 * 7,  # ResNet output flattened
            visual_embedding_dim,
        )

        # Multimodal encoder
        self.multimodal_encoder = VisualBertModel.from_pretrained(
            "uclanlp/visualbert-vqa-coco-pre"
        )

        # Heads
        self.obj_head = nn.Linear(hidden_dim, 2)
        self.att_head = nn.Linear(hidden_dim, 2)
        self.rel_head = nn.Linear(hidden_dim, 2)
        self.sce_head = nn.Linear(hidden_dim, 2)
        self.oth_head = nn.Linear(hidden_dim, 2)
        self.all_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        embeddings: torch.Tensor,        # (B, T, 4096)
        attention_masks: torch.Tensor,   # (B, T)
        images: List[Image.Image],
        is_all: bool = False,
    ) -> torch.Tensor:

        # =======================
        # check
        # =======================
        assert embeddings.shape[-1] == self.proj.in_features, \
            f"Expected embedding dim {self.proj.in_features}, got {embeddings.shape[-1]}"

        device = embeddings.device

        visual_inputs = self.image_processor(images, return_tensors="pt")
        visual_inputs = {k: v.to(device) for k, v in visual_inputs.items()}

        visual_outputs = self.vision_encoder(**visual_inputs)

        # (B, 2048, 7, 7)
        visual_last_hidden_states = visual_outputs.last_hidden_state

        B = visual_last_hidden_states.size(0)

        # Flatten spatial dims
        visual_flat = visual_last_hidden_states.view(B, -1)  # (B, 2048*7*7)

        # Project → (B, 512)
        visual_embeddings = self.vision_proj(visual_flat)

        # Add sequence dimension → (B, 1, 512)
        visual_embeddings = visual_embeddings.unsqueeze(1)

        # Masks
        visual_token_type_ids = torch.ones(
            (B, self.img_seq_len),
            dtype=torch.long,
            device=device
        )

        visual_attention_mask = torch.ones(
            (B, self.img_seq_len),
            dtype=torch.long,
            device=device
        )

        proj_embeddings = self.proj(embeddings.float())  # (B, T, 768)

        out = self.multimodal_encoder(
            inputs_embeds=proj_embeddings,
            attention_mask=attention_masks,
            visual_embeds=visual_embeddings,
            visual_token_type_ids=visual_token_type_ids,
            visual_attention_mask=visual_attention_mask,
        )

        out_embeddings = out.last_hidden_state  # (B, T+1, 768)

        # Remove visual token (last one)
        out_embeddings = out_embeddings[:, :-1, :]  # (B, T, 768)

        if is_all:
            return self.all_head(out_embeddings)

        obj_logits = self.obj_head(out_embeddings)
        att_logits = self.att_head(out_embeddings)
        rel_logits = self.rel_head(out_embeddings)
        sce_logits = self.sce_head(out_embeddings)
        oth_logits = self.oth_head(out_embeddings)

        return obj_logits, att_logits, rel_logits, sce_logits, oth_logits