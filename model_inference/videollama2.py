import os, sys
sys.path.append('VideoLLaMA2')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from utils.video_process import download_video
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
import json
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
import hashlib
import requests
from tqdm import tqdm
import jsonlines

def get_videollama2(model_name):
    model, processor, tokenizer = model_init(model_name)
    return model, processor, tokenizer

def generate_single_mcq_response(model, processor, tokenizer, video_path, qa_text_prompt, modal='video'):
    # 处理单轮问答
    input_data = {
        "prompt": qa_text_prompt,
        "multi_modal_data": {"video": video_path}
    }
    response = mm_infer(
        processor[modal](input_data['multi_modal_data']['video']),
        input_data['prompt'],
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        modal=modal
    )
    return response

def generate_by_videollama2_single_mcq(model_name, 
                                     queries, 
                                     prompt,
                                     output_path,
                                     total_frames, 
                                     temperature, 
                                     max_tokens):
    model, processor, tokenizer = get_videollama2(model_name)
    
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
                    video_summary=query['video_summary'],
                    qa_dict=qa_dict,
                    round=1,
                    prompt=prompt
                )
                reasoning = generate_single_mcq_response(model, processor, tokenizer, video_path, qa_text_prompt1)
                
                # 第二轮生成最终答案
                _, qa_text_prompt2 = prepare_qa_text_input(
                    model_name=model_name,
                    video_summary=query['video_summary'],
                    qa_dict=qa_dict,
                    round=2,
                    prompt=prompt
                )
                final_answer = generate_single_mcq_response(model, processor, tokenizer, video_path, qa_text_prompt2)
                
                output_dict["single_mcq_result"][qa] = {
                    "reasoning_steps_and_answer": reasoning,
                    "mcq_answer": final_answer
                }
            
            f.write(output_dict)

def generate_by_videollama2_multi_mcq(model_name, 
                                    queries, 
                                    prompt,
                                    output_path,
                                    total_frames, 
                                    temperature, 
                                    max_tokens):
    model, processor, tokenizer = get_videollama2(model_name)
    
    with jsonlines.open(output_path, 'a') as f:
        for query in tqdm(queries):
            output_dict = {
                "video_id": query['video_id'],
                "multi_mcq_result": {}
            }
            video_path, _ = download_video(query['video_path'])
            
            for qa, qa_dict in query['multi_mcq'].items():
                _, qa_text_prompt = prepare_qa_text_input(
                    model_name=model_name,
                    video_summary=query['video_summary'],
                    qa_dict=qa_dict,
                    round=1,
                    prompt=prompt
                )
                
                # 处理多轮问答
                qa_text_prompts = [qa_text_prompt] if isinstance(qa_text_prompt, str) else qa_text_prompt
                for idx, prompt_text in enumerate(qa_text_prompts):
                    response = generate_single_mcq_response(model, processor, tokenizer, video_path, prompt_text)
                    output_dict["multi_mcq_result"][f"{qa}_{idx+1}"] = response
            
            f.write(output_dict)

def generate_response(model_name: str, 
                    prompt: dict,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    if prompt["type"] == "single_mcq":
        generate_by_videollama2_single_mcq(
            model_name=model_name,
            queries=queries,
            prompt=prompt,
            output_path=output_path,
            total_frames=total_frames,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    elif prompt["type"] == "multi_mcq":
        generate_by_videollama2_multi_mcq(
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