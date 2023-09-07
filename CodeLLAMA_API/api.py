from fastapi import FastAPI, Request
import uvicorn, json, datetime
from transformers import AutoTokenizer
import transformers
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global pipeline, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = int(json_post_list.get('max_length'))
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    system = "\nProvide answers in Python"
    user = "Write a function that {}".format(json_post_list.get('prompt'))
    prompt = f"<s><<SYS>>\n{system}\n<</SYS>>\n\n{user}"
    #prompt = json_post_list.get('prompt') + system

    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=1,
        temperature=0.1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
        add_special_tokens=False
        )

    response = ""
    for seq in sequences:
        #response += f"Result: {seq['generated_text']}"
        response += seq['generated_text']

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'

    torch_gc()
    return answer


if __name__ == '__main__':

    tmodel = "CodeLlama-7b-Instruct-hf/"
    #tmodel = "CodeLlama-7b-Python-hf/"
    tokenizer = AutoTokenizer.from_pretrained(tmodel)
    pipeline = transformers.pipeline(
        "text-generation",
        model=tmodel,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
