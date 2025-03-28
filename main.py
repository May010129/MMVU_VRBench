from tqdm import tqdm
import json
import argparse
import os
import sys
from utils.constant import COT_PROMPT, DO_PROMPT, SINGLE_ROUND_MCQ, MULTI_ROUND_MCQ
from utils.video_process import download_video
from transformers.utils import logging
logging.set_verbosity_error() 

def main(
    model_name: str, 
    prompt: str, 
    queries: list, 
    total_frames: int, 
    output_path: str, 
    n: int=1)-> None:
    if "gpt-4o" in model_name:
        from model_inference.azure_gpt import generate_response
    elif "gemini" in model_name or "grok" in model_name:
        from model_inference.openai_compatible import generate_response
    elif "glm-4v" in model_name:
        from model_inference.glm4v import generate_response
    elif "claude" in model_name:
        from model_inference.claude import generate_response
    elif model_name in json.load(open("model_inference/vllm_model_list.json")):
        from model_inference.vllm_video_inference import generate_response
    elif "InternVideo2_5" in model_name:
        from model_inference.internvideo2_5 import generate_response
    elif "InternVideo2" in model_name:
        from model_inference.internvideo import generate_response
    elif "VideoLLaMA2" in model_name:
        from model_inference.videollama2 import generate_response
    elif "VideoChat" in model_name:
        from model_inference.videochat import generate_response
    elif "VideoLLaMA3" in model_name:
        from model_inference.videollama3 import generate_response
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    generate_response(model_name=model_name,
                    prompt=prompt,
                    queries=queries, 
                    total_frames=total_frames, 
                    output_path=output_path,
                    n = n)
        
prompt_dict = {
    "single_round": SINGLE_ROUND_MCQ,
    "multi_round": MULTI_ROUND_MCQ,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--prompt', type=str, default="single_round")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--total_frames', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument("--api_base",type=str,default="")
    parser.add_argument("--n",type=int,default=1)

    args = parser.parse_args()
    
    model_name = args.model
    total_frames = args.total_frames 

    try:
        prompt = prompt_dict[args.prompt]
    except KeyError:
        print("Invalid prompt")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
        
    output_dir = os.path.join(args.output_dir, f"{args.prompt}")
    os.makedirs(output_dir, exist_ok=True)
    
    output_name = model_name.split("/")[-1]
    if total_frames == -1:
        output_path = os.path.join(output_dir, f"{output_name}_1fps.jsonl")
    else:
        output_path = os.path.join(output_dir, f"{output_name}_{total_frames}frame.jsonl")

    total_json_ls = [json.loads(line) for line in open(args.data_path, "r")]
    
    if os.path.exists(output_path):
        orig_output = [json.loads(line) for line in open(output_path, "r")]
        exist_id = set([long_vid['video_id'] for long_vid in orig_output])

    total_json_ls = [line for line in total_json_ls if line['video_id'] not in exist_id]

    print(f"=========Running {args.model}=========\n")
    
    queries = []
    key = "one_step_mcq" if args.prompt == "single_round" else "multi_step_mcq"
    for long_vid in total_json_ls:
        query = {
                "video_id": long_vid['video_id'],
                "video_path": long_vid['video_path'],
                "video_summary": long_vid['video_summary'],
                key: long_vid[key]
            }
        queries.append(query)
    
    #TODO
    # multi-gpu inference

    main(
        model_name = args.model, 
        prompt = prompt, 
        queries = queries, 
        total_frames = total_frames, 
        output_path = output_path, 
        n = args.n)