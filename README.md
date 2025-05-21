# 3D-LLMs

## Описание папок
Папка 'abc_dataset' содержит данные из ABC датасета (https://archive.nyu.edu/handle/2451/44309);

Папка 'example_material' содержит следующее:
   - папка 'collages_3', 'collages_4', 'collages_6' содержит коллажи картинок, состоящие из 3/4/6 изображений соответственно, созданные в результате применения Isomap
   - папка 'rendered_imgs' содержит многоракурсные изображения детали
   - файл 'example_object_path.pkl' содержит массив всех файлов формата .obj

## Файл find_parts.py

Файл **find_parts.py** содержит функцию, которая отбирает из данных датасета ABC только те модели, которые состоят из одной детали.
   - 5.111 из 10.000 моделей состоят из одной детали; 
   - Список подходящих файлов формата OBJ сохраняется в файл example_object_path.pkl.

## Файлы render_script_type1.py и render_script_type2.py

Файлы render_script_type1.py и render_script_type2.py реализует рендеринг 3D-модели в 28 изображений.
   - Предварительно надо установить blender (https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip) и разместить в корне проекта
   - За основу взят код из статьи Сap3D. Ссылка на оригинальный код: https://github.com/tiangeluo/DiffuRank
   - Команды для запуска скриптов:
```
# Создание 8 ракурсов
blender -b -P render_script_type1.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'

# Создание 20 ракурсов
blender -b -P render_script_type2.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'
```

## Файл dim_reduction_solved_3d_model.ipynb

Файл **dim_reduction_solved_3d_model.ipynb** реализует выбор трех различных фотографий модели с разных ракурсов и их объединение в единное изображение.
   - Нужно определиться, как будет реализован выбор трех картинок.
