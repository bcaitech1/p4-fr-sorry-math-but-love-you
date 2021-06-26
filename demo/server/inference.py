import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

import sys
sys.path.append("/opt/ml/code")

from utils.flags import Flags
from utils.utils import id_to_string_for_serve
from data.augmentations import get_valid_transforms
from postprocessing.postprocessing import get_decoding_manager
from networks.EfficientSATRN import EfficientSATRN_for_serve
# from networks.LiteSATRN import LiteSATRN


def prepare_model():
    checkpoint_path = "/opt/ml/code/models/satrn-fold-2-0.8171.pth"  # for EfficientSATRN
    # checkpoint_path = "/opt/ml/code/models/LiteSATRN_best_model.pth"  # for LiteSATRN
    if torch.cuda.is_available():
        device = torch.device("cuda")
        checkpoint = torch.load(checkpoint_path)
    else:
        device = torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    
    options = Flags(checkpoint["configs"]).get()

    decoding_manager = False
    manager = (
        get_decoding_manager(
            tokens_path="/opt/ml/input/data/train_dataset/tokens.txt", batch_size=batch_size
        )
        if decoding_manager
        else None
    )

    model = EfficientSATRN_for_serve(options, checkpoint, decoding_manager).to(device)  # for EfficientSATRN
    # model = LiteSATRN(options, checkpoint, decoding_manager).to(device)  # for LiteSATRN
    model.eval()

    return model, device, checkpoint, options


def inference(model, device, checkpoint, options, image):
    transforms = get_valid_transforms(
        height=options.input_size.height, width=options.input_size.width
    )
    # image = Image.open(image)  # for test
    image = image.convert("RGB")
    w, h = image.size
    if h / w > 2:
        image = image.rotate(90, expand=True)
    image = np.array(image)
    image = transforms(image=image)["image"]

    with torch.no_grad():
        input = image.float().to(device)
        output = model(input)
        decoded_values = output.transpose(1, 2)  # [B, VOCAB_SIZE, MAX_LEN]
        _, sequence = torch.topk(decoded_values, 1, dim=1)  # sequence: [B, 1, MAX_LEN]
        sequence = sequence.squeeze(1)
        sequence_str = id_to_string_for_serve(sequence, checkpoint, do_eval=1)

        return sequence_str
