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
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
from utils.video_process import download_video
import hashlib
import requests
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


import jsonlines

# Keep all the existing helper functions (build_transform, find_closest_aspect_ratio, 
# dynamic_preprocess, load_image, get_index, get_num_frames_by_duration, load_video)
# unchanged since they're utility functions

def get_internvideo2_5(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    return model, tokenizer

def create_input(qa_text_prompt, tokenizer, model, video_path, max_num_frames=512):
    pixel_values, num_patches_list = load_video(video_path, num_segments=max_num_frames, max_num=1, get_frame_by_duration=False)
    pixel_values = pixel_values.half().to(model.device)
    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
    question = video_prefix + qa_text_prompt
    return pixel_values, num_patches_list, question

def generate_response(model, tokenizer, pixel_values, question, num_patches_list):
    generation_config = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1024,
        top_p=0.1,
        num_beams=1
    )
    response = model.chat(tokenizer, pixel_values, question, generation_config, 
                         num_patches_list=num_patches_list, history=None, return_history=False)
    return response

def generate_by_internvideo2_5_single_mcq(model_name, 
                                        queries, 
                                        prompt,
                                        output_path,
                                        total_frames, 
                                        temperature, 
                                        max_tokens):
    model, tokenizer = get_internvideo2_5(model_name)
    max_num_frames = 512
    
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
                
                pixel_values, num_patches_list, question = create_input(
                    qa_text_prompt, tokenizer, model, video_path, max_num_frames
                )
                response1 = generate_response(model, tokenizer, pixel_values, question, num_patches_list)
                output_dict["single_mcq_result"][qa] = {'reasoning_steps_and_answer': response1}
                
                # Second round for final answer
                _, qa_text_prompt_2 = prepare_qa_text_input(
                    model_name=model_name,
                    video_summary=query['video_summary'],
                    qa_dict=qa_dict,
                    round=2,
                    prompt=prompt
                )
                
                # For second round, we need to include history
                history = [(question, response1)]
                generation_config = dict(
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=1024,
                    top_p=0.1,
                    num_beams=1
                )
                response2 = model.chat(tokenizer, pixel_values, qa_text_prompt_2, generation_config, 
                                     num_patches_list=num_patches_list, history=history, return_history=False)
                
                output_dict["single_mcq_result"][qa]['mcq_answer'] = response2
            
            f.write(output_dict)

def generate_by_internvideo2_5_multi_mcq(model_name, 
                                       queries, 
                                       prompt,
                                       output_path,
                                       total_frames, 
                                       temperature, 
                                       max_tokens):
    model, tokenizer = get_internvideo2_5(model_name)
    max_num_frames = 512
    
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
                    pixel_values, num_patches_list, question = create_input(
                        prompt_text, tokenizer, model, video_path, max_num_frames
                    )
                    response = generate_response(model, tokenizer, pixel_values, question, num_patches_list)
                    output_dict["multi_mcq_result"][f"{qa}_{idx+1}"] = response
            
            f.write(output_dict)

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    if prompt["type"] == "single_mcq":
        generate_by_internvideo2_5_single_mcq(
            model_name, 
            queries, 
            prompt=prompt, 
            output_path=output_path,
            total_frames=total_frames, 
            temperature=GENERATION_TEMPERATURE, 
            max_tokens=MAX_TOKENS
        )
    elif prompt['type'] == "multi_mcq":
        generate_by_internvideo2_5_multi_mcq(
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