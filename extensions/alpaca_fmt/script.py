import gradio as gr
import modules.shared as shared
# from modules.chat import clean_chat_message
from modules.extensions import apply_extensions
from modules.text_generation import encode, get_max_prompt_length

alpaca_fmt = """### Instruction:
{}

### Response:"""


def custom_generate_chat_prompt(user_input, max_new_tokens, name1, name2, context, chat_prompt_size, impersonate=False):
    print(f'{user_input=}')
    print(f'{context=}')
    # user_input = clean_chat_message(user_input)
    rows = [f"{context.strip()}\n"]
    rows = []
    if shared.soft_prompt:
        chat_prompt_size -= shared.soft_prompt_tensor.shape[1]
    max_length = min(get_max_prompt_length(max_new_tokens), chat_prompt_size)

    i = 0
    while i < len(shared.history['internal']) and len(encode(''.join(rows), max_new_tokens)[0]) < max_length:
        rows.insert(1, f"{name2}: {shared.history['internal'][i][1].strip()}\n")
        if not (shared.history['internal'][i][0] == '<|BEGIN-VISIBLE-CHAT|>'):
            rows.insert(1, f"{name1}: {shared.history['internal'][i][0].strip()}\n")
        i += 1

    if not impersonate:
        rows.append(f"{name1}: {user_input}\n")
        rows.append(apply_extensions(f"{name2}:", "bot_prefix"))
        limit = 1
    else:
        rows.append(f"{name1}:")
        limit = 2

    while len(rows) > limit and len(encode(''.join(rows), max_new_tokens)[0]) >= max_length:
        rows.pop(1)

    prompt = ''.join(rows)
    print(f'{prompt=}')
    fmt_prompt = alpaca_fmt.format(
        prompt)
    print(f'{fmt_prompt=}')
    return fmt_prompt


def ui():
    pass
