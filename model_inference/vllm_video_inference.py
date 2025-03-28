from utils.vlm_prepare_input import *
import json
import jsonlines
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS

model_map = {
    "llava-hf/LLaVA-NeXT-Video-7B-hf": prepare_llava_next_video,
    "llava-hf/LLaVA-NeXT-Video-34B-hf": prepare_llava_next_video,
    "Qwen/Qwen2-VL-7B-Instruct": prepare_qwen2, 
    "Qwen/Qwen2-VL-2B-Instruct": prepare_qwen2,
    "Qwen/Qwen2-VL-72B-Instruct-AWQ": prepare_qwen2, 
    "microsoft/Phi-3.5-vision-instruct": prepare_phi3v, 
    "OpenGVLab/InternVL2-8B": prepare_general_vlm,
    "OpenGVLab/InternVL2_5-78B-AWQ": prepare_general_vlm,
    "OpenGVLab/InternVL2_5-8B": prepare_general_vlm,
    "OpenGVLab/InternVL2_5-38B":
    prepare_general_vlm,
    "mistral-community/pixtral-12b": prepare_pixtral,
    "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": prepare_llava_onevision,
    "unsloth/Llama-3.2-11B-Vision-Instruct": prepare_mllama,
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit": prepare_mllama,
    "h2oai/h2ovl-mississippi-2b": prepare_general_vlm,
    "nvidia/NVLM-D-72B": prepare_general_vlm,
    "HuggingFaceM4/Idefics3-8B-Llama3": prepare_general_vlm,
    "deepseek-ai/deepseek-vl2": prepare_deepseek_vl2,
    "deepseek-ai/deepseek-vl2-tiny": prepare_deepseek_vl2,
    "deepseek-ai/deepseek-vl2-small": prepare_deepseek_vl2,
    "rhymes-ai/Aria-Chat": prepare_aria,
    "Qwen/Qwen2.5-VL-3B-Instruct": prepare_qwen2,
    "Qwen/Qwen2.5-VL-7B-Instruct": prepare_qwen2,
}

model_input_map = {
    "llava-hf/LLaVA-NeXT-Video-7B-hf": prepare_llava_next_video_inputs,
    "llava-hf/LLaVA-NeXT-Video-34B-hf": prepare_llava_next_video_inputs,
    "Qwen/Qwen2-VL-7B-Instruct": prepare_qwen2_inputs, 
    "Qwen/Qwen2-VL-2B-Instruct": prepare_qwen2_inputs,
    "Qwen/Qwen2-VL-72B-Instruct-AWQ": prepare_qwen2_inputs, 
    "microsoft/Phi-3.5-vision-instruct": prepare_phi3v_inputs, 
    "OpenGVLab/InternVL2-8B": prepare_general_vlm_inputs,
    "OpenGVLab/InternVL2_5-78B-AWQ": prepare_general_vlm_inputs,
    "OpenGVLab/InternVL2_5-8B": prepare_general_vlm_inputs,
    "OpenGVLab/InternVL2_5-38B":
    prepare_general_vlm_inputs,
    "mistral-community/pixtral-12b": prepare_pixtral_inputs,
    "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": prepare_llava_onevision_inputs,
    "unsloth/Llama-3.2-11B-Vision-Instruct": prepare_mllama_inputs,
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit": prepare_mllama_inputs,
    "h2oai/h2ovl-mississippi-2b": prepare_general_vlm_inputs,
    "nvidia/NVLM-D-72B": prepare_general_vlm_inputs,
    "HuggingFaceM4/Idefics3-8B-Llama3": prepare_general_vlm_inputs,
    "deepseek-ai/deepseek-vl2": prepare_deepseek_vl2_inputs,
    "deepseek-ai/deepseek-vl2-tiny": prepare_deepseek_vl2_inputs,
    "deepseek-ai/deepseek-vl2-small": prepare_deepseek_vl2_inputs,
    "rhymes-ai/Aria-Chat": prepare_aria_inputs,
    "Qwen/Qwen2.5-VL-3B-Instruct": prepare_qwen2_inputs,
    "Qwen/Qwen2.5-VL-7B-Instruct": prepare_qwen2_inputs,
}

def generate_vlm_response_single_mcq(model_name, 
                       queries, 
                       total_frames, 
                       prompt,
                       output_path, 
                       temperature: float=1, 
                       max_tokens: int=1024):
    llm, sampling_params = model_map[model_name](model_name, total_frames, temperature, max_tokens)
    
    with jsonlines.open(output_path, 'a') as f:
        for query in queries:
            qa_id, inputs = model_input_map[model_name](model_name, query, prompt, round=1, total_frames=total_frames)
            output_dict = {"video_id":query['video_id'], "single_mcq_result":{}}
            assert len(qa_id) == len(inputs)
            for idx, input in enumerate(inputs):
                responses = llm.generate(input, sampling_params=sampling_params)
                responses = [response.outputs[0].text for response in responses][0]
                output_dict["single_mcq_result"][qa_id[idx]]['reasoning_steps_and_answer'] = responses
            inputs = model_input_map[model_name](model_name, query, prompt, 2, inputs, responses, total_frames, temperature, max_tokens)
            for idx, input in enumerate(inputs):
                responses = llm.generate(input, sampling_params=sampling_params)
                responses = [response.outputs[0].text for response in responses][0]
                output_dict["single_mcq_result"][qa_id[idx]]['mcq_answer'] = responses
            f.write(output_dict)

def generate_vlm_response_multi_mcq(model_name, 
                       queries, 
                       total_frames, 
                       prompt,
                       output_path, 
                       temperature: float=1, 
                       max_tokens: int=1024):
    llm, sampling_params = model_map[model_name](model_name, queries, prompt, total_frames, temperature, max_tokens)
    
    with jsonlines.open(output_path, 'a') as f:
        for query in queries:
            video_id, qa_id, inputs = model_input_map[model_name](model_name, query, prompt, 1, inputs, responses, total_frames, temperature, max_tokens)
            output_dict = {"video_id":video_id, "multi_mcq_result":{}}
            assert len(qa_id) == len(inputs)
            for idx, input in enumerate(inputs):
                responses = llm.generate(input, sampling_params=sampling_params)
                responses = [response.outputs[0].text for response in responses][0]
                output_dict["multi_mcq_result"][qa_id[idx]]= responses
            f.write(output_dict)


def generate_response(model_name: str,          
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1,
                    temperature: float=GENERATION_TEMPERATURE,
                    max_tokens: int=MAX_TOKENS):
    if model_name not in model_map:
        raise ValueError(f"Model type {model_name} is not supported.")
    
    # judge if it is single_roud or multi_round
    if prompt["type"] == "single_mcq":    
        generate_vlm_response_single_mcq(
                            model_name, 
                            queries, 
                            prompt=prompt, 
                            total_frames=total_frames, 
                            output_path=output_path,
                            temperature=temperature, 
                            max_tokens=max_tokens)  
    elif prompt['type'] == "multi_mcq":
        generate_vlm_response_multi_mcq(
                            model_name, 
                            queries, 
                            prompt=prompt, 
                            total_frames=total_frames, 
                            output_path=output_path,
                            temperature=temperature, 
                            max_tokens=max_tokens)  
    else:
        raise ValueError(f"prompt type is not supported.")