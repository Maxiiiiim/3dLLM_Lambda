import os
import pickle
import json
import base64
from base64 import b64encode
from openai import OpenAI
from mistralai import Mistral

prompt_path3 = 'example_material/prompts/prompt_3.txt'
prompt_path4 = 'example_material/prompts/prompt_4.txt'
prompt_path6 = 'example_material/prompts/prompt_6.txt'

# Функция для кодирования изображения в base64
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_response_from_image_mistral(image_path, prompt, api_key):
    model = "pixtral-12b-2409"
    base64_image = encode_image_to_base64(image_path)

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ]
    )
    return response.choices[0].message.content


def generate_response_from_image_qwen(image_path, model_name, prompt, my_api_key):
    client = OpenAI(
        api_key= my_api_key, # DASHSCOPE_API_KEY
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image_to_base64(image_path)

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user","content": [
                {"type": "text","text": prompt},
                {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}]
        )
    return completion.choices[0].message.content


def run_mistral(api_key, image_path, prompt):
    try:
        response = generate_response_from_image_mistral(image_path, prompt, api_key)
        # print(response)
        return response
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


def run_qwen(api_key, model_name, image_path, prompt):
    try:
        response = generate_response_from_image_qwen(image_path, model_name, prompt, api_key)
        # print(response)
        return response
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


def create_json_with_mistral(object_dirpath, prompt_path, output_path):
    json_data = {}
    api_key_mistral = "MISTRAL_API_KEY"

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt = file.read()

    for dirpath, dirnames, filenames in os.walk(object_dirpath):
        for file_path in filenames:
          path = os.path.join(dirpath, file_path)
          part_number = file_path.split(".")[0]

          response = run_mistral(api_key_mistral, path, prompt)
          json_data[part_number] = response

    with open(output_path, 'wb') as f:
        pickle.dump(json_data, f)

def create_json_with_qwen(api_key, model, object_dirpath, prompt_path, output_path):
    json_data = {}

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt = file.read()

    for dirpath, dirnames, filenames in os.walk(object_dirpath):
        for file_path in filenames:
          path = os.path.join(dirpath, file_path)
          part_number = file_path.split(".")[0]

          response = run_qwen(api_key, model, path, prompt)
          json_data[part_number] = response

    with open(output_path, 'wb') as f:
        pickle.dump(json_data, f)


object_path3, object_path4, object_path6 = '3', '4', '6'

mistral_pkl_path3 = './results/json_responses/json_mistral_3.pkl'
mistral_pkl_path4 = './results/json_responses/json_mistral_4.pkl'
mistral_pkl_path6 = './results/json_responses/json_mistral_6.pkl'

create_json_with_mistral(object_path3, prompt_path3, mistral_pkl_path3)
create_json_with_mistral(object_path4, prompt_path4, mistral_pkl_path4)
create_json_with_mistral(object_path6, prompt_path6, mistral_pkl_path6)

api_key_qwen = "DASHSCOPE_API_KEY"

vl_max_pkl_path3 = './results/json_responses/json_qwen_vl_max_3.pkl'
vl_max_pkl_path4 = './results/json_responses/json_qwen_vl_max_4.pkl'
vl_max_pkl_path6 = './results/json_responses/json_qwen_vl_max_6.pkl'

create_json_with_qwen(api_key_qwen, "qwen-vl-max", object_path3, prompt_path3, vl_max_pkl_path3)
create_json_with_qwen(api_key_qwen, "qwen-vl-max", object_path4, prompt_path4, vl_max_pkl_path4)
create_json_with_qwen(api_key_qwen, "qwen-vl-max", object_path6, prompt_path6, vl_max_pkl_path6)

qwen_72b_pkl_path3 = './results/json_responses/json_qwen2_vl_72b_instruct_3.pkl'
qwen_72b_pkl_path4 = './results/json_responses/json_qwen2_vl_72b_instruct_4.pkl'
qwen_72b_pkl_path6 = './results/json_responses/json_qwen2_vl_72b_instruct_6.pkl'

create_json_with_qwen(api_key_qwen, "qwen2.5-vl-72b-instruct", object_path3, prompt_path3, qwen_72b_pkl_path3)
create_json_with_qwen(api_key_qwen, "qwen2.5-vl-72b-instruct", object_path4, prompt_path4, qwen_72b_pkl_path4)
create_json_with_qwen(api_key_qwen, "qwen2.5-vl-72b-instruct", object_path6, prompt_path6, qwen_72b_pkl_path6)


def save_jsons(json_pkl_paths, json_collages_paths):
    for i in range(len(json_pkl_paths)):
        json_data = pickle.load(open(json_pkl_paths[i], 'rb'))
        for key, value in json_data.items():
            cleaned_json = value.strip().replace('json', '').replace('```', '').strip()
            json_data = json.loads(cleaned_json)
            output_path = os.path.join(json_collages_paths[i], f"{key}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)


qwen_72b_pkl_paths = [qwen_72b_pkl_path3, qwen_72b_pkl_path4, qwen_72b_pkl_path6]
qwen_72b_json_paths = ['./results/json_responses/qwen2_5_vl_72b/collages_3',
                       './results/json_responses/qwen2_5_vl_72b/collages_4',
                       './results/json_responses/qwen2_5_vl_72b/collages_6']
save_jsons(qwen_72b_pkl_paths, qwen_72b_json_paths)

qwen_vl_max_pkl_paths = [vl_max_pkl_path3, vl_max_pkl_path4, vl_max_pkl_path6]
qwen_vl_max_json_paths = ['./results/json_responses/qwen_vl_max/collages_3',
                          './results/json_responses/qwen_vl_max/collages_4',
                          './results/json_responses/qwen_vl_max/collages_6']
save_jsons(qwen_vl_max_pkl_paths, qwen_vl_max_json_paths)

mistral_pkl_paths = [mistral_pkl_path3, mistral_pkl_path4, mistral_pkl_path6]
mistral_json_paths = ['./results/json_responses/pixtral_12b/collages_3',
                      './results/json_responses/pixtral_12b/collages_4',
                      './results/json_responses/pixtral_12b/collages_6']
save_jsons(mistral_pkl_paths, mistral_json_paths)