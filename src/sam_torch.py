import os
import time

import requests
import segment_anything
import torch

import benchmark

HUGE_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
HUGE_BUILD = segment_anything.build_sam_vit_h
HUGE_LOCAL = "/tmp/sam_h.pth"
LARGE_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
LARGE_BUILD = segment_anything.build_sam_vit_l
LARGE_LOCAL = "/tmp/sam_l.pth"
BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
BASE_BUILD = segment_anything.build_sam_vit_b
BASE_LOCAL = "/tmp/sam_b.pth"

URL = BASE_URL
LOCAL = BASE_LOCAL
build_sam = BASE_BUILD

def download_file(url, local_filename):
    if not os.path.exists(local_filename):
        os.makedirs(os.path.dirname(local_filename), exist_ok=True)
        print(f"Downloading {url}...")
        response = requests.get(url)
        with open(local_filename, "wb") as file:
            file.write(response.content)
        print(f"Download complete: {local_filename}")
    else:
        print(f"File already exists: {local_filename}")


def get_model():
    download_file(URL, LOCAL)
    return build_sam(checkpoint=LOCAL).cuda()


def get_dataset():
    input_image = torch.Tensor(benchmark.SAM_BATCH_SIZE, 3, 1024, 1024).cuda()
    input_point = torch.Tensor([[[500, 375], [250, 375]]]).cuda()
    input_label = torch.Tensor([[1, 2]]).cuda()
    y_true = torch.Tensor(benchmark.SAM_BATCH_SIZE, 256, 64, 64).cuda()
    return input_image, input_point, input_label, y_true


@torch.no_grad
def inference(model, input_image, input_point, input_label):
    features = model.image_encoder(input_image)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=(input_point, input_label), boxes=None, masks=None
    )
    return model.mask_decoder(
        image_embeddings=features,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )


def train(model, input_image, y_true):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    optimizer.zero_grad()
    y_pred = model(input_image)
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    optimizer.step()

    start_time = time.time()
    for _ in range(benchmark.NUM_STEPS):
        optimizer.zero_grad()
        y_pred = model(input_image)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
    end_time = time.time()

    return (end_time - start_time) / benchmark.NUM_STEPS * 1000

def run():
    model = get_model()
    input_image, input_point, input_label, y_true = get_dataset()

    training_time = train(model.image_encoder, input_image, y_true)

    inference(model, input_image, input_point, input_label)
    start_time = time.time()
    for i in range(benchmark.NUM_STEPS):
        inference(model, input_image, input_point, input_label)
    end_time = time.time()
    inference_time = (end_time - start_time) / benchmark.NUM_STEPS * 1000
    return training_time, inference_time


if __name__ == "__main__":
    benchmark.benchmark(run)
