import os
import pickle
import json
import openai
from base64 import b64encode

OPENAI_API_KEY = "openai-api-key"

object_path3 = './example_material/collages_3'
object_path4 = './example_material/collages_4'
object_path6 = './example_material/collages_6'

prompt_path3 = 'example_material/prompts/prompt_3.txt'
prompt_path4 = 'example_material/prompts/prompt_4.txt'
prompt_path6 = 'example_material/prompts/prompt_6.txt'

pkl_path3 = './example_material/json_standard/json_standard_3.pkl'
pkl_path4 = './example_material/json_standard/json_standard_4.pkl'
pkl_path6 = './example_material/json_standard/json_standard_6.pkl'

def generate_response_from_image(image_path, prompt, api_key):
    """
    Генерирует ответ модели на основе изображения и текстового промпта

    :param image_path: путь к изображению
    :param prompt: текстовый промпт
    :param api_key: OpenAI API ключ
    :return: ответ модели
    """
    # Проверяем, что файл существует
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл {image_path} не найден")

    # Кодируем изображение в base64
    with open(image_path, "rb") as image_file:
        base64_image = b64encode(image_file.read()).decode('utf-8')

    # Устанавливаем API ключ
    openai.api_key = api_key

    try:
        # Отправляем запрос к API
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,  # Максимальное количество токенов в ответе
        )

        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Ошибка при обращении к OpenAI API: {str(e)}")

# Пример использования
def run_code(OPENAI_API_KEY, image_path, prompt):
    try:
        response = generate_response_from_image(image_path, prompt, OPENAI_API_KEY)
        print(response)
        return response
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

def create_json_benchmarks(object_dirpath, prompt_path, output_path):
    json_data = {}
    prompt = ""

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt = file.read()

    for dirpath, dirnames, filenames in os.walk(object_dirpath):
        for file_path in filenames:
          path = os.path.join(dirpath, file_path)
          part_number = file_path.split(".")[0]

          response = run_code(OPENAI_API_KEY, path, prompt)
          json_data[part_number] = response

    with open(output_path, 'wb') as f:
        pickle.dump(json_data, f)


# create_json_benchmarks(object_path3, prompt_path3, output_path3)
# create_json_benchmarks(object_path4, prompt_path4, output_path4)
# create_json_benchmarks(object_path6, prompt_path6, output_path6)

# uid_paths = pickle.load(open('./example_material/json_standard/json_standard_6.pkl', 'rb'))
# print(len(uid_paths))
# for k, v in uid_paths.items():
#     print(k)
#     print(v)

json_pkl_paths = [pkl_path3, pkl_path4, pkl_path6]
json_collages_paths = ['./example_material/json_standard/json_collages_3', './example_material/json_standard/json_collages_4', './example_material/json_standard/json_collages_6']
for i in range(len(json_pkl_paths)):
    json_data = pickle.load(open(json_pkl_paths[i], 'rb'))
    for key, value in json_data.items():
        cleaned_json = value.strip().replace('json', '').replace('```', '').strip()
        json_data = json.loads(cleaned_json)
        output_path = os.path.join(json_collages_paths[i], f"{key}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)