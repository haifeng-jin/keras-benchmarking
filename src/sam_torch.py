import os
import time

import requests
import segment_anything
import torch

import benchmark


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
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    local_filename = "/tmp/sam_vit_h_4b8939.pth"
    download_file(url, local_filename)
    return segment_anything.build_sam_vit_h(checkpoint=local_filename).cuda()


def get_dataset():
    input_image = torch.Tensor(benchmark.SAM_BATCH_SIZE, 3, 1024, 1024).cuda()
    input_point = torch.Tensor([[[500, 375], [250, 375]]]).cuda()
    input_label = torch.Tensor([[1, 2]]).cuda()
    return input_image, input_point, input_label


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


def run():
    model = get_model()
    input_image, input_point, input_label = get_dataset()
    inference(model, input_image, input_point, input_label)
    start_time = time.time()
    for i in range(benchmark.NUM_STEPS):
        inference(model, input_image, input_point, input_label)
    end_time = time.time()
    return None, (end_time - start_time) / benchmark.NUM_STEPS * 1000


if __name__ == "__main__":
    benchmark.benchmark(run)
