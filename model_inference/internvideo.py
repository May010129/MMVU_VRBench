import os, torch
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
import json
from transformers import AutoTokenizer, AutoModel
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import decord
from decord import VideoReader, cpu
from torchvision import transforms
decord.bridge.set_bridge("torch")
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
from utils.video_process import download_video
import hashlib
import requests
from tqdm import tqdm

import jsonlines

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video_path, num_segments=16, return_msg=False, resolution=224, hd_num=4, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)
    frames = transform(frames)

    T_, C, H, W = frames.shape
        
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames

def get_internvideo(model_name):
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    return model, tokenizer

def create_input(qa_text_prompt, video_path):
    text_input = f"{qa_text_prompt}"
    video_tensor = load_video(video_path, num_segments=16, return_msg=False)
    
    input = {
        "prompt": text_input,
        "multi_modal_data": {
            "video": video_tensor
        },
    }
    return input

def generate_response(input, model, tokenizer):
    input['multi_modal_data']['video'] = input['multi_modal_data']['video'].to(model.device)
    response = model.chat(
        tokenizer, 
        '', 
        input["prompt"], 
        media_type='video', 
        media_tensor=input['multi_modal_data']['video'], 
        generation_config={'do_sample': False}
    )
    return response

def generate_by_internvideo_single_mcq(model_name, 
                                     queries, 
                                     prompt,
                                     output_path,
                                     total_frames, 
                                     temperature, 
                                     max_tokens):
    
    model, tokenizer = get_internvideo(model_name)
    
    with jsonlines.open(output_path, 'a') as f:
        for idx, query in enumerate(tqdm(queries)):
            output_dict = {"video_id": query['video_id'], "single_mcq_result": {}}
            video_path, _ = download_video(query['video_path'])
            
            for qa, qa_dict in query['single_mcq'].items():
                _, qa_text_prompt = prepare_qa_text_input(
                    model_name=model_name,
                    video_summary=query['video_summary'],
                    qa_dict=qa_dict,
                    round=1,
                    prompt=prompt
                )
                
                input = create_input(qa_text_prompt, video_path)
                response1 = generate_response(input, model, tokenizer)
                output_dict["single_mcq_result"][qa] = {'reasoning_steps_and_answer': response1}
                
                # Second round for final answer
                _, qa_text_prompt_2 = prepare_qa_text_input(
                    model_name=model_name,
                    video_summary=query['video_summary'],
                    qa_dict=qa_dict,
                    round=2,
                    prompt=prompt
                )
                
                input = create_input(qa_text_prompt_2, video_path)
                response2 = generate_response(input, model, tokenizer)
                output_dict["single_mcq_result"][qa]['mcq_answer'] = response2
                
            f.write(output_dict)

def generate_by_internvideo_multi_mcq(model_name, 
                                    queries, 
                                    prompt,
                                    output_path,
                                    total_frames, 
                                    temperature, 
                                    max_tokens):
    
    model, tokenizer = get_internvideo(model_name)
    
    with jsonlines.open(output_path, 'a') as f:
        for idx, query in enumerate(tqdm(queries)):
            output_dict = {"video_id": query['video_id'], "multi_mcq_result": {}}
            video_path, _ = download_video(query['video_path'])
            
            for qa, qa_dict in query['multi_mcq'].items():
                _, qa_text_prompt = prepare_qa_text_input(
                    model_name=model_name,
                    video_summary=query['video_summary'],
                    qa_dict=qa_dict,
                    round=1,
                    prompt=prompt
                )
                
                qa_text_prompt = [qa_text_prompt] if isinstance(qa_text_prompt, str) else qa_text_prompt
                for idx, prompt_text in enumerate(qa_text_prompt):
                    input = create_input(prompt_text, video_path)
                    response = generate_response(input, model, tokenizer)
                    output_dict["multi_mcq_result"][f"{qa}_{idx+1}"] = response
                    
            f.write(output_dict)

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    
    assert model_name in ["OpenGVLab/InternVideo2-Chat-8B"], "Invalid model name"
    
    if prompt["type"] == "single_mcq":
        generate_by_internvideo_single_mcq(
            model_name, 
            queries, 
            prompt=prompt,
            output_path=output_path,
            total_frames=total_frames,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    elif prompt['type'] == "multi_mcq":
        generate_by_internvideo_multi_mcq(
            model_name, 
            queries, 
            prompt=prompt,
            output_path=output_path,
            total_frames=total_frames,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    else:
        raise ValueError(f"prompt type is not supported.")