from keybert.llm import TextGeneration
from keybert import KeyLLM

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from fastapi import FastAPI
from pydantic import BaseModel

model_name = "TheBloke/Llama-2-7B-Chat-GPTQ"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

keyword_generator = pipeline(
    model=model,
    tokenizer=tokenizer,
    task='text-generation',
    max_new_tokens=8,
    repetition_penalty=1.1
)

feedback_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.5,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

keyword_example_prompt = """
<s>[INST]
I have the following document:
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
[/INST] meat, beef, eat, eating, emissions, steak, food, health, processed, chicken</s>"""

keyword_ins_prompt = """
[INST]
I have the following document:
- [DOCUMENT]

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
[/INST]
"""

keyword_prompt = keyword_example_prompt + keyword_ins_prompt

key_llm = TextGeneration(keyword_generator, prompt=keyword_prompt)
kw_model = KeyLLM(key_llm)

def get_missing_keywords(response, expected):
    response_keywords = kw_model.extract_keywords(response)[0]
    expected_keywords = kw_model.extract_keywords(expected)[0]

    return list(set(expected_keywords) - set(response_keywords))

def get_feedback(question, response, expected):

    prompt = f'''
[INST]
<<SYS>>
You are a teacher and you are grading a student's response to a question.
Here is an example of what you should do:
Question: "What is the capital of France?"
Response: "Lyon"
Expected: "Paris"
Feedback: "The student has confused Lyon and Paris. Lyon is the second largest city in France, but Paris is the capital."
<</SYS>>
Now, you are grading the following response:
Question: "{question}"
Response: "{response}"
Expected: "{expected}"

Give feedback to the student on their response. Make sure to be specific and constructive. Just give feedback on the response, not the question or anything else.
Wrap your feedback in [FEEDBACK] and [/FEEDBACK] tags.
[/INST]
'''

    return feedback_generator(prompt)[0]['generated_text']
    

app = FastAPI()

class APIBody(BaseModel):
    question: str
    response: str
    expected: str

@app.get("/")
def generate_keywords(apiBody: APIBody):

    question = apiBody.question
    response = apiBody.response
    expected = apiBody.expected

    return {
        "missing_keywords": get_missing_keywords(response,expected),
        "feedback": get_feedback(question, response, expected),
    }
