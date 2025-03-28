from utils.video_process import download_video
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
import json
from utils.prepare_input import prepare_qa_text_input
from tqdm import tqdm
import jsonlines

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

def get_videollama3(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor

def create_input(qa_text_prompt, processor, video_path):
    text_input = f"{qa_text_prompt}"
            
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 180}},
                {"type": "text", "text": text_input},
            ]
        },
    ]

    inputs = processor(
        conversation=conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    return conversation, inputs
    
def generate_response(input, model, processor):
    output_ids = model.generate(
                **input,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                max_new_tokens=2048,
            )
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

def generate_by_videollama3_single_mcq(model_name, 
                            queries, 
                            prompt,
                            output_path,
                            total_frames, 
                            temperature, 
                            max_tokens):

    model, processor = get_videollama3(model_name)
    with jsonlines.open(output_path, 'a') as f:
        for idx, query in enumerate(tqdm(queries)):
            output_dict = {"video_id":query['video_id'], "single_mcq_result":{}}
            video_path, _ = download_video(query['video_path']) 
            for qa, qa_dict in query['single_mcq'].items():
                _, qa_text_prompt = prepare_qa_text_input(
                        model_name=model_name,
                        video_summary=query['video_summary'],
                        qa_dict=qa_dict,
                        round=1,
                        prompt=prompt
                    )
                conversation, input = create_input(qa_text_prompt, processor, video_path)
                response1 = generate_response(input, model, processor)
                output_dict["single_mcq_result"][qa]['reasoning_steps_and_answer'] = response1
                conversation.append({
                    "role":"assistant",
                    "content": [
                        {"type": "text", "text": response1},
                    ]
                })
                _, qa_text_prompt_2 = prepare_qa_text_input(
                        model_name=model_name,
                        video_summary=query['video_summary'],
                        qa_dict=qa_dict,
                        round=2,
                        prompt=prompt
                    )
                conversation.append({
                    "role":"user",
                    "content": [
                        {"type": "text", "text": qa_text_prompt_2},
                    ]
                })
                input = processor(
                    conversation=conversation,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                input = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in input.items()}
                if "pixel_values" in input:
                    input["pixel_values"] = input["pixel_values"].to(torch.bfloat16)
                response2 = generate_response(input, model, processor)
                output_dict["single_mcq_result"][qa]['mcq_answer'] = response2
            f.write(output_dict)
    
def generate_by_videollama3_multi_mcq(model_name, 
                            queries, 
                            prompt,
                            output_path,
                            total_frames, 
                            temperature, 
                            max_tokens):

    model, processor = get_videollama3(model_name)

    with jsonlines.open(output_path, 'a') as f:
        for idx, query in enumerate(tqdm(queries)):
            output_dict = {"video_id":query['video_id'], "multi_mcq_result":{}}
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
                for idx, prompt in enumerate(qa_text_prompt):
                    conversation, input = create_input(prompt, processor, video_path)
                    response = generate_response(input, model, processor)
                    output_dict["multi_mcq_result"][f"{qa}_{idx+1}"] = response
            f.write(output_dict)

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    if prompt["type"] == "single_mcq":  
        generate_by_videollama3_single_mcq(model_name, 
                    queries, 
                    prompt=prompt, 
                    output_path=output_path,
                    total_frames=total_frames, 
                    temperature=GENERATION_TEMPERATURE, 
                    max_tokens=MAX_TOKENS)
    elif prompt['type'] == "multi_mcq":
        generate_by_videollama3_multi_mcq(model_name, 
                    queries, 
                    prompt=prompt, 
                    output_path=output_path,
                    total_frames=total_frames, 
                    temperature=GENERATION_TEMPERATURE, 
                    max_tokens=MAX_TOKENS)
    else:
        raise ValueError(f"prompt type is not supported.")