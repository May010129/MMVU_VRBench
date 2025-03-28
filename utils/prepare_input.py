from utils.video_process import read_video, download_video, prepare_base64frames, prepare_gemini_video_input, prepare_base64_video
from utils.constant import COT_PROMPT
import google.generativeai as genai
import requests
import os
import time
from tqdm import tqdm
import hashlib
import base64

def dict_to_text(question, options):
    option_prompt = "\n".join(f"{k}: {v}" for k, v in options.items())
    return f"Question: {question}\nOptions:\n{option_prompt}"

def get_previous_reasoning(idx, previous_steps):
    """Generate reasoning text from previous steps."""
    if idx == 0:
        return ""
    return "\n".join(
        f"question: {step['question']}\nanswer: {step['options'][step['correct']]}"
        for step in previous_steps
    )


def prepare_qa_text_input(video_summary, qa_dict, round, prompt):
    if prompt["type"] == "single_mcq":
        if round == 1:
            qa_text_prompt = prompt["content"]["1st-round"].substitute(
                question=qa_dict["question"],
                video_summary=video_summary
            )
            return {"type": "text", "text": qa_text_prompt}, qa_text_prompt
            
        elif round == 2:
            return prompt["content"]["2nd-round"].substitute(
                multiple_choice_question=dict_to_text(qa_dict["question"], qa_dict["options"])
            )
    elif prompt["type"] == "multi_mcq":
        mcq_data = qa_dict["mcq_data"]
        if qa_dict["reasoning_type"] == "Event Summarization":
            qa_text_prompt = prompt["content"].substitute(
                previous_reasoning="",
                question=dict_to_text(mcq_data["summary_mcq"]["question"], mcq_data["summary_mcq"]["options"]),
                video_summary=video_summary
            )
            return {"type": "text", "text": qa_text_prompt}, qa_text_prompt
        else:
            messages, prompts = [], []
            for idx, step in enumerate(mcq_data["steps"]):
                qa_text_prompt = prompt["content"].substitute(
                    previous_reasoning=get_previous_reasoning(idx, mcq_data["steps"][:idx]),
                    question=dict_to_text(step["question"], step["options"]),
                    video_summary=video_summary
                )
                messages.append({"type": "text", "text": qa_text_prompt})
                prompts.append(qa_text_prompt)
            return messages, prompts
    else:
        raise ValueError(f"Invalid question type: {prompt["type"]}")

def prepare_multi_image_input(model_name, video_path, total_frames, video_tmp_dir = "video_cache"):
    base64frames = prepare_base64frames(model_name, video_path, total_frames, video_tmp_dir = video_tmp_dir)

    if "claude" in model_name:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame,
                },
            } for frame in base64frames
        ]
    # for vllm models
    elif "/" in model_name: 
        return base64frames
    else:
        return [
            {
                "type": "image_url",
                'image_url': {
                    "url": f"data:image/jpeg;base64,{frame}",
                },
            } for frame in base64frames
        ]


def prepare_qa_inputs(model_name, queries, total_frames, prompt=COT_PROMPT):
    messages = []
    for query in tqdm(queries):
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        if total_frames >= 1:
            vision_input = prepare_multi_image_input(model_name, query['video'], total_frames)

            prompt_message = [
                {
                    "role": "user",
                    "content": vision_input + [qa_text_message],
                },
            ]
        elif total_frames == 0:
            prompt_message = [
                {
                    "role": "user",
                    "content": [qa_text_message],
                },
            ]
        elif total_frames == -1:
            if "gemini" in model_name:
                video_file = prepare_gemini_video_input(query['video'])
                prompt_message = [video_file, qa_text_prompt]
            elif model_name in ["glm-4v-plus-0111","glm-4v-plus", "glm-4v"]:
                video_url = query['video']
                base64_video = prepare_base64_video(video_url)
                prompt_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": qa_text_prompt},
                            {"type": "video_url", "video_url": {"url": base64_video}}
                        ] 
                    }
                ]
            else:
                raise ValueError(f"Invalid model name: {model_name}")

        messages.append(prompt_message)
    return messages
