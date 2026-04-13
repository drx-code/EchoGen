import torch
import numpy as np

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, get_phrases_from_posmap

GROUNDING_DINO_CONFIG = "tools/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "pretrained/groundingdino/groundingdino_swint_ogc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_grounding_dino():
    return load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT)


dino_model = load_grounding_dino()

def load_image(image_rgb: str):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(image_rgb)
    image_transformed, _ = transform(image_rgb, None)
    return image, image_transformed

def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, device='cpu', token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)


    return boxes_filt, pred_phrases

def SubjectSegmentation(image_rgb, text_prompt="subject", device='cuda'):
    image_rgb, image = load_image(image_rgb)
    H, W, _ = image_rgb.shape

    boxes, phrases = get_grounding_output(
        model=dino_model,
        image=image,
        caption=text_prompt,
        box_threshold=0.01,
        text_threshold=0.25,
        device=device
    )

    if boxes.shape[0] == 0:
        return

    boxes = boxes * torch.tensor([W, H, W, H])
    box = boxes[0]
    box[:2] -= box[2:] / 2
    box[2:] += box[:2]
    box = box.cpu().numpy().astype(int)
    
    xx, yy = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    x_min, y_min, x_max, y_max = box

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W, x_max)
    y_max = min(H, y_max)
    mask = (xx >= x_min) & (xx < x_max) & (yy >= y_min) & (yy < y_max)  # (H, W)

    mask = np.repeat(mask[:,:,None], 3, axis=-1)  # (C, H, W)

    result = np.full_like(image_rgb, 255)
    result[mask] = image_rgb[mask]
    return result
