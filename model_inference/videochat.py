import os, torch
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
import json
import jsonlines
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


def get_videochat_flash(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    return model, tokenizer

def generate_by_videochat_flash_single_mcq(model_name, queries, prompt, output_path, total_frames, temperature, max_tokens):
    model, tokenizer = get_videochat_flash(model_name)
    max_num_frames = 512
    media_dict = {'video_read_type': 'decord'}
    
    with jsonlines.open(output_path, 'a') as f:
        for query in tqdm(queries):
            output_dict = {
                "video_id": query['video_id'],
                "single_mcq_result": {}
            }
            video_path, _ = download_video(query['video_path'])
            
            for qa, qa_dict in query['single_mcq'].items():
                # 第一轮生成推理步骤
                _, qa_text_prompt1 = prepare_qa_text_input(
                    model_name=model_name,
                    video_summary=query.get('video_summary', ''),
                    qa_dict=qa_dict,
                    round=1,
                    prompt=prompt
                )
                response1 = model.chat(
                    video_path,
                    tokenizer,
                    qa_text_prompt1,
                    chat_history=None,
                    return_history=False,
                    max_num_frames=max_num_frames,
                    media_dict=media_dict,
                    generation_config={
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "do_sample": False,
                        "top_p": None,
                        "num_beams": 1
                    }
                )
                
                # 第二轮生成最终答案
                _, qa_text_prompt2 = prepare_qa_text_input(
                    model_name=model_name,
                    video_summary=query.get('video_summary', ''),
                    qa_dict=qa_dict,
                    round=2,
                    prompt=prompt
                )
                # 传入对话历史（假设历史格式为[用户输入，模型响应]）
                chat_history = [(qa_text_prompt1, response1)]
                response2 = model.chat(
                    video_path,
                    tokenizer,
                    qa_text_prompt2,
                    chat_history=chat_history,
                    return_history=False,
                    max_num_frames=max_num_frames,
                    media_dict=media_dict,
                    generation_config={
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "do_sample": False,
                        "top_p": None,
                        "num_beams": 1
                    }
                )
                
                # 保存结果
                output_dict["single_mcq_result"][qa] = {
                    'reasoning_steps_and_answer': response1,
                    'mcq_answer': response2
                }
            
            f.write(output_dict)

def generate_by_videochat_flash_multi_mcq(model_name, queries, prompt, output_path, total_frames, temperature, max_tokens):
    model, tokenizer = get_videochat_flash(model_name)
    max_num_frames = 512
    media_dict = {'video_read_type': 'decord'}
    
    with jsonlines.open(output_path, 'a') as f:
        for query in tqdm(queries):
            output_dict = {
                "video_id": query['video_id'],
                "multi_mcq_result": {}
            }
            video_path, _ = download_video(query['video_path'])
            
            for qa, qa_dict in query['multi_mcq'].items():
                _, qa_text_prompts = prepare_qa_text_input(
                    model_name=model_name,
                    video_summary=query.get('video_summary', ''),
                    qa_dict=qa_dict,
                    round=1,
                    prompt=prompt
                )
                if isinstance(qa_text_prompts, str):
                    qa_text_prompts = [qa_text_prompts]
                
                for idx, prompt_text in enumerate(qa_text_prompts):
                    response = model.chat(
                        video_path,
                        tokenizer,
                        prompt_text,
                        chat_history=None,
                        return_history=False,
                        max_num_frames=max_num_frames,
                        media_dict=media_dict,
                        generation_config={
                            "max_new_tokens": max_tokens,
                            "temperature": temperature,
                            "do_sample": False,
                            "top_p": None,
                            "num_beams": 1
                        }
                    )
                    output_dict["multi_mcq_result"][f"{qa}_{idx+1}"] = response
            
            f.write(output_dict)

def generate_response(model_name: str, 
                    prompt: dict,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    # 根据提示类型分发处理逻辑
    if prompt["type"] == "single_mcq":
        generate_by_videochat_flash_single_mcq(
            model_name=model_name,
            queries=queries,
            prompt=prompt,
            output_path=output_path,
            total_frames=total_frames,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    elif prompt["type"] == "multi_mcq":
        generate_by_videochat_flash_multi_mcq(
            model_name=model_name,
            queries=queries,
            prompt=prompt,
            output_path=output_path,
            total_frames=total_frames,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    else:
        raise ValueError(f"Prompt type is not supported.")
