# Streaming with Fine-Tuning model and RAG

import os, gc, sys, torch, json, subprocess

import gradio as gr

from threading      import Thread
from peft           import PeftConfig, get_peft_model, IA3Config
from transformers   import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from langchain.vectorstores import FAISS
from langchain.embeddings   import HuggingFaceEmbeddings
from langchain              import PromptTemplate

##### Init #####

### Loading custom settings ###
HUGGINGFACEHUB_API_TOKEN = {HUGGINGFACE_API_KEY}
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

embeddings = HuggingFaceEmbeddings(model_name = "jhgan/ko-sroberta-multitask")

### Load FAISS vector data ###
dbr_embed_chunks = FAISS.load_local({VECTOR_DATASET_DIRECTORY}, embeddings)

### Local LLM ###
peft_model_id = {MODEL_DIRECTORY}
config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, 
    device_map = "auto", 
    load_in_8bit = True,
    torch_dtype = torch.float16,
    rope_scaling = {"type": "dynamic", "factor": 2.0}
    )
model = get_peft_model(model, config)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, return_token_type_ids = False)
tokenizer.pad_token = tokenizer.eos_token

##### Server #####
last_query_text = ""

def get_gpu_info(nvidia_smi_path = "nvidia-smi", no_units = True):
    
    DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
    )
    
    nu_opt = '' if not no_units else ',nounits'
    
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % ('nvidia-smi', ','.join(DEFAULT_ATTRIBUTES), nu_opt)
    
    output = subprocess.check_output(cmd, shell = True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]
    
    # gpu info
    gpu0_max_memory = int(lines[0].split(',')[4].strip())
    gpu1_max_memory = int(lines[1].split(',')[4].strip())

    gpu0_allocated_memory = int(lines[0].split(',')[6].strip())
    gpu1_allocated_memory = int(lines[1].split(',')[6].strip())

    total_alloacated_memory = gpu0_allocated_memory + gpu1_allocated_memory
    gpu_max_memory = gpu0_max_memory + gpu1_max_memory

    total_allocated_percentage = round(gpu_max_memory / total_alloacated_memory, 2)
    
    return total_allocated_percentage

def semantic_search_with_score (query, use_score=***, ref_score=***):
    use_docs = []
    ref_docs = []
    i = 0

    docs_and_scores = dbr_embed_chunks.similarity_search_with_score(query, k=10)
    
    for doc, score in docs_and_scores:
        if score <= use_score:
            use_docs.append(docs_and_scores[i])
        elif score <= ref_score:
            ref_docs.append(docs_and_scores[i])

        i += 1

    return use_docs, ref_docs
    
def print_references(user_message):
    
    if len(user_message) == 0:
        global last_query_text
        user_message = last_query_text
    
    article = dbr_embed_chunks.similarity_search_with_score(user_message, k = 5)
    
    doc_cnt = len(article)
    top1_score = article[0][-1]
    
    if top1_score >= ***:
        
        reference_log = ""
        
        return reference_log
    
    else:
        
        trg_string = "***"
        
        reference_log = [
            {
                "url" : f"***/{article[i][0].metadata['source'].split('|')[0]}" if trg_string not in article[i][0].metadata['source'] and article[i][0].metadata['source'].split(' |')[0] != "None" else "",
                "title" : article[i][0].metadata['source'].split('|')[4].strip().replace("&#8226;", "·") if trg_string not in article[i][0].metadata['source'] else "",
                "issue" : article[i][0].metadata['source'].split('|')[1].strip().split(' ')[0] if trg_string not in article[i][0].metadata['source'] else '|'.join(article[i][0].metadata['source'].split('|')[2:]).strip(),
                "pub_date" : article[i][0].metadata['source'].split('|')[1].strip().split(' ')[1].replace('(', '').replace(')', '') if trg_string not in article[i][0].metadata['source'] else ""
            } for i in range(doc_cnt)
        ]
        reference_log = json.dumps(reference_log, ensure_ascii = False)
                
        return reference_log


##### UI #####

def generate_by_llm_only(user_message, history):
    if not user_message:
        print('Empty Input')
        return ""
    else:
        global last_query_text
        last_query_text = user_message

    # RAG
    with torch.no_grad():
        
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            use_docs, ref_docs = semantic_search_with_score(user_message)
            similarity_score = use_docs[0][-1]
            
            if similarity_score > ***:
                prompt = "***"
                prompt = PromptTemplate.from_template(prompt)
                prompt = prompt.format(query = user_message)
                
                # print("##### prompt check : ", prompt)
                
                ## stream 
                model_inputs = tokenizer.encode(prompt, return_tensors = 'pt').to('cuda')
                streamer = TextIteratorStreamer(tokenizer, timeout = 100, skip_prompt = True, skip_special_tokens = True)

                generate_kwargs = dict(
                    inputs              = model_inputs,
                    streamer            = streamer,
                    temperature         = ***,
                    top_p               = ***,
                    top_k               = ***,
                    max_new_tokens      = ***,
                    repetition_penalty  = ***,
                    do_sample           = ***,
                    use_cache           = ***,
                    early_stopping      = ***,
                    num_beams           = ***,
                    eos_token_id        = ***,
                    pad_token_id        = ***,
                )

                t = Thread(target = model.generate, kwargs = generate_kwargs)
                t.start()

                model_output = ''
                for new_text in streamer:
                    model_output += new_text
                    yield model_output
                
                gc.collect()
                torch.cuda.empty_cache()
                
            else:
                RAG_context = use_docs[0][0].page_content
                
                # Prompt Templates
                prompt = """
                ***
                """
                prompt = PromptTemplate.from_template(prompt)
                prompt = prompt.format(query = user_message, context = RAG_context)
            
                # print("##### prompt check : ", prompt)
            
                ## stream 
                model_inputs = tokenizer.encode(prompt, return_tensors = 'pt').to('cuda')
                streamer = TextIteratorStreamer(tokenizer, timeout = 100, skip_prompt = True, skip_special_tokens = True)

                generate_kwargs = dict(
                    inputs              = model_inputs,
                    streamer            = streamer,
                    temperature         = ***,
                    top_p               = ***,
                    top_k               = ***,
                    max_new_tokens      = ***,
                    repetition_penalty  = ***,
                    do_sample           = ***,
                    use_cache           = ***,
                    early_stopping      = ***,
                    num_beams           = ***,
                    eos_token_id        = ***,
                    pad_token_id        = ***,
                )

                t = Thread(target = model.generate, kwargs = generate_kwargs)
                t.start()

                model_output = ''
                for new_text in streamer:
                    model_output += new_text
                    yield model_output
                    
                gc.collect()
                torch.cuda.empty_cache()
            
        except IndexError as ie:
            model_output = "해당 질문의 내용은 지식베이스에 없는 사항으로 모델을 더 고도화하여 추후에 답변하도록 하겠습니다."
            yield model_output
    

def clear_chat():
    return [], []

def process_example(args):
    for [x, y] in generate(args):
        pass
    return [x, y]

##### UI Variables #####

examples = [
    "우리 회사에서 올해부터 OKR을 본격적으로 도입하라는 지시가 내려왔어. 그런데 기존 성과 관리 시스템인 MBO(Management by Objective)와 OKR은 어떻게 다른거야?",
    "착한 디자인이 무엇인지 설명해주세요?",
    "로봇을 도입해 공장 업무를 자동화해보려고 하는데 기존 직원들이 적응에 어려움을 겪거나 반발하지는 않을까?",
    "시리즈 A 단계의 식품 스타트업이 진출할 만한 해외 국가 추천해 줘.",
    "스타트업 직원 복지 사례를 알려줘. 그리고 이들 스타트업은 왜 복지에 힘을 쏟는 걸까?",
]

title = """<h1 align="center">💬📰 AskBiz-Demo Playground 📰💬</h1>"""

title2 = """<div style="text-align: center; max-width: 500px; margin: 0 auto;">
                <div>
                    <h1>AskBiz Prototype</h1>
                </div>
                        <p style="margin-bottom: 10px; font-size: 94%">
                            developed by <a href="http://bigster.co.kr/">Bigster</a>
                        </p>
                </div>"""

custom_css = """
#banner-image {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
#chat-message {
    font-size: 14px;
    min-height: 300px;
}
"""

if __name__ == "__main__":

    with gr.Blocks(analytics_enabled = False, css = custom_css) as demo:
        gr.HTML(title)

        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "🤗 AskBiz Streaming on gradio\n"
                    "& Current of AskBiz is a demo that showcases the model fine-tuned for LoRA"
                )

        with gr.Row():
            with gr.Box():
                output = gr.Markdown()
                # chatbot = gr.ChatBot(elem_id = 'chat-message', label = 'Chat')
                chatIf = gr.ChatInterface(generate_by_llm_only, examples=examples).queue()

        with gr.Row():
            with gr.Column(scale = 3):
                user_message = gr.Textbox(elem_id = 'q-input', show_label = False, visible = False,
                                          placeholder = "AskBiz에게 물어보고싶은 것을 입력해주세요!")

                with gr.Row():
                    with gr.Column(scale=1, min_width=600):
                        reference_show_btn = gr.Button("참고 내용 자세히 보기", elem_id = 'show-reference-btn')
                        reference_log = gr.Textbox(elem_id='reference_log', label='* 대답에 사용된 기사 및 추가 참고 내용에 대한 정보')

                with gr.Accordion(label = "Parameters", open = False, elem_id = "h-parameters-accordion"):
                    temperature = gr.Slider(
                        label = 'Temperature',
                        value = ***,
                        minimum = 0.0,
                        maximum = 1.0,
                        step = 0.1,
                        interactive = True,
                        info = "높은 값을 설정하면 더 다양하고 창의적인 결과값을 만들어냅니다"
                    )
                    top_p = gr.Slider(
                        label = 'Top-p (nuclues sampling)',
                        value = ***,
                        minimum = 0.0,
                        maximum = 1,
                        step = 0.05,
                        interactive = True,
                        info = "높은 값을 설정하면 더 낮은 결과값을 가지는 토큰을 샘플링합니다"
                    )
                    top_k = gr.Slider(
                        label = 'Top-k sampling',
                        value = ***,
                        minimum = 0,
                        maximum = ***,
                        step = 1,
                        interactive = True,
                        info = "상위 k개의 단어 중에 여러 단어가 동일한 확률 값을 가지면 이들 중에서 무작위로 선택합니다"
                    )
                    max_new_tokens = gr.Slider(
                        label = 'Max new tokens',
                        value = ***,
                        minimum = ***,
                        maximum = ***,
                        step = 4,
                        interactive = True,
                        info = "문장의 길이를 조정할 수 있습니다"
                    )
                    repetition_penalty = gr.Slider(
                        label = 'Repetition Penalty',
                        value = ***,
                        minimum = 0.0,
                        maximum = 10,
                        step = 0.1,
                        interactive = True,
                        info = "단어의 등장을 제어하는 데 사용되며 동일한 단어를 반복하지 않거나 최소화하도록 조정하는 역할을 수행합니다"
                    )
                    
                with gr.Row():
                    gr.Markdown(
                        "Askbiz-Demo가 생산하는 결과는 정확한 정보를 포함하고 있지 않을 수 있으며 이전 맥락을 기억하지 못합니다. "
                        "Askbiz-Demo model엔 동아비즈니스리뷰(DBR)의 기사 데이터셋이 사용되었습니다. ", 
                        elem_classes = ['disclaimer']
                    )
                    

        history = gr.State([])
        last_user_message = gr.State("")

        reference_show_btn.click(fn = print_references, inputs = user_message, outputs = reference_log)

    # demo.queue(concurrency_count = 16).launch(server_name = "0.0.0.0", share = True, server_port = 8081)
    demo.queue().launch(share=False, debug=False, ssl_verify=False, server_name="0.0.0.0", ssl_certfile="cert.pem", ssl_keyfile="key.pem", server_port = 7860)