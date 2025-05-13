import pickle
import yaml
import os

def find_models_1part():
    # Архив скачан отсюда: https://archive.nyu.edu/handle/2451/44309
    statistics_path = "C:\\Users\\mminecz\\PycharmProjects\\PythonProject\\3dLLM_Lambda\\abc_dataset\\statistics"
    object_path = "C:\\Users\\mminecz\\PycharmProjects\\PythonProject\\3dLLM_Lambda\\abc_dataset\\object"
    parts = []
    obj_files =[]

    # Проходим по всем подпапкам и файлам
    for dirpath, dirnames, filenames in os.walk(statistics_path):
        for filename in filenames:
            if filename.endswith(".yaml") or filename.endswith(".yml"):  # можно убрать, если нужны все файлы
                file_path = os.path.join(dirpath, filename)
                part_number = dirpath.split(sep='\\')[-1]

                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        data = yaml.safe_load(file)
                        if data['#parts'] == 1:
                            parts.append(part_number)
                    except Exception as e:
                        print(f"Ошибка при чтении {file_path}: {e}")

    # Проходим по всем подпапкам и файлам
    for dirpath, dirnames, filenames in os.walk(object_path):
        for filename in filenames:
            if filename.endswith(".obj"):  # можно убрать, если нужны все файлы
                file_path = os.path.join(dirpath, filename)
                part_number = dirpath.split(sep='\\')[-1]

                if part_number in parts:
                    obj_files.append(os.path.join(f"./abc_dataset/object/{part_number}/", filename))

                # with open(file_path, 'r', encoding='utf-8') as file:
                #     try:
                #         data = yaml.safe_load(file)
                #         if part_number in parts:
                #             obj_files.append(part_number)
                #     except Exception as e:
                #         print(f"Ошибка при чтении {file_path}: {e}")

    return obj_files

object_files = find_models_1part()
with open('./example_material/example_object_path.pkl', 'wb') as f:
    pickle.dump(object_files, f)
print(object_files)