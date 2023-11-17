import argparse, warnings, torch
warnings.filterwarnings(action = 'ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, get_peft_model, PeftModel



def main(
    base_model_id: str,
    lora_model_id: str,
    output_dir: str
    ):
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map = 'auto')
    
    try:
        config = PeftConfig.from_pretrained(lora_model_id)
        lora_model = get_peft_model(base_model, config)
        
    except RuntimeError as re:
        lora_model = PeftModel.from_pretrained(
            base_model, 
            lora_model_id, 
            device_map = {"" : "cpu"})
    
    base_vocab_size = base_model.get_input_embeddings().weight.size(0)
    
    print(f"Base Model vocab size : {base_vocab_size}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    if base_vocab_size != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"Resizing Vocabulary to {len(tokenizer)}")
        
    base_first_weight = base_model.gpt_neox.layers[0].attention.query_key_value.weight.clone()
    lora_first_weight = lora_model.gpt_neox.layers[0].attention.query_key_value.weight.clone()
    
    assert torch.allclose(base_first_weight, lora_first_weight)
    
    
    standalone_model = lora_model.merge_and_unload()
    standalone_model.save_pretrained(output_dir, safe_serialization = True)
    tokenizer.save_pretrained(output_dir)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument('--base_model_id',
                        type = 'str',
                        default = "beomi/KoAlpaca-Polyglot-12.8B")
    parser.add_argument('--lora_model_id',
                        type = str,
                        default = "finetuning/")
    parser.add_argument('--output_dir',
                        type = str,
                        default = "finietuning_standalone/")
    
    args = parser.parse_args()
    
    main(
        base_model_id = args.base_model_id,
        peft_model_id = args.lora_model_id,
        output_dir = args.output_dir
    )