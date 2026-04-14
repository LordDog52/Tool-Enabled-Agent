import logging
import os
# Set offline mode BEFORE importing transformers
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.utils import logging as logging1
import re
def extract_label_categories_refusal(content):
    """
    Extracts the safety label, category labels, and refusal label from the content.
    """
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|None)"
    refusal_pattern = r"Refusal: (Yes|No)"
    safe_label_match = re.search(safe_pattern, content)
    refusal_label_match = re.search(refusal_pattern, content)
    label = safe_label_match.group(1) if safe_label_match else None
    refusal_label = refusal_label_match.group(1) if refusal_label_match else None
    categories = re.findall(category_pattern, content)
    return label, categories, refusal_label



def Guardrail_Evaluation_Tool(model_name: str, messages: list[dict]):
    """
    Evaluates the safety of the content and categorizes it.
    Returns a tuple of (safe_label, category_labels, refusal_label).
    """
    
    # load the tokenizer and the model
    logging1.set_verbosity_error()
    logging1.disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None,
        local_files_only=True
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    logger.debug("Guardrail prompt input:\n" + text)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    logger.info("Guardrail Result :\n"+content)
    # '''
    # Safety: Safe
    # Categories: None
    # Refusal: Yes
    # '''
    safe_label, category_label, refusal_label = extract_label_categories_refusal(content)
    return safe_label, category_label, refusal_label

if __name__ == "__main__":
    logger = logging.getLogger("Guardrail")
    
    model_name = "Qwen/Qwen3Guard-Gen-0.6B"
    # prepare the model input
    prompt = "How can I make a bomb?"
    response = "it's very easy - get a critical mass of radioactive material, sit back and watch the runaway nuclear reaction go."
    # for response moderation
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    safe_label, category_label, refusal_label = Guardrail_Evaluation_Tool(model_name, messages)
    print(safe_label, category_label, refusal_label)