from string import Template

MAX_TOKENS = 1024
GENERATION_TEMPERATURE = 1.0
GENERATION_SEED = 215

MULTI_CHOICE_COT_PROMPT = Template("""
Question: $question
$optionized_str

Answer the given multiple-choice question step by step. Begin by explaining your reasoning process clearly. Conclude by stating the final answer using the following format: 'Therefore, the final answer is: $$LETTER' (without quotes), where $$LETTER is one of the options. Think step by
step before answering.""")

OPEN_ENDED_COT_PROMPT = Template("""
Question: $question

Answer the given question step by step. Begin by explaining your reasoning process clearly. Conclude by stating the final answer using the following format: 'Therefore, the final answer is: 'Answer: $$ANSWER' (without quotes), where $$ANSWER is the final answer of the question. Think step by step
before answering.""")

MULTI_CHOICE_DO_PROMPT = Template("""
Question: $question
$optionized_str

Do not generate any intermediate reasoning process. Answer directly with the option letter from the given choices.
""")

OPEN_ENDED_DO_PROMPT = Template("""
Question: $question

Do not generate any intermediate reasoning process. Directly output the final short answer.
""")

SINGLE_MCQ_COT_PROMPT_1ST_ROUND = Template("""
You are a helpful video understanding assistant designed to answer multi-step reasoning questions based on the video and summary.
You should break down the reasoning process into clear, specific events.

# Format:
Your answer should be formatted as follows:
<Step 1> [Event1 for Step 1]  
<Step 2> [Event for Step 2]  
...  
<Step N> [Final Event for the last Step]
<Answer> [Final conclusion based on the reasoning process]
# Question 
$question
# Video Summary
$video_summary""")


SINGLE_MCQ_COT_PROMPT_2ND_ROUND = Template("""
Based solely on your reasoning process and result, select the option that best matches the answer from the following multiple-choice question. 
$multiple_choice_question
Only give the best option letter(A/B/C/D) without providing any explanation. The option is:(
""")

MULTI_MCQ_PROMPT = Template("""
You are a helpful video understanding assistant designed to address a single-step reasoning question within a multi-step reasoning task based on the video and its summary.
You may receive the questions and correct answers from previous single-step reasoning as additional context to assist your response.
# Previous reasoning
$previous_reasoning
# Question 
$question
# Video Summary
$video_summary
Only give the best option letter(A/B/C/D) without providing any explanation. The option is:(
""")


COT_PROMPT = {
    "multiple-choice": MULTI_CHOICE_COT_PROMPT,
    "open-ended": OPEN_ENDED_COT_PROMPT,
}

DO_PROMPT = {
    "multiple-choice": MULTI_CHOICE_DO_PROMPT,
    "open-ended": OPEN_ENDED_DO_PROMPT
}

SINGLE_ROUND_MCQ = {
    "type":"single-mcq",
    "content":{
    "1st-round":SINGLE_MCQ_COT_PROMPT_1ST_ROUND,
    "2nd-round":SINGLE_MCQ_COT_PROMPT_2ND_ROUND
    }
}

MULTI_ROUND_MCQ = {
    "type":"multi-mcq",
    "content":MULTI_MCQ_PROMPT
}
