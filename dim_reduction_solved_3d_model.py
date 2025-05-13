import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from matplotlib import offsetbox
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

import warnings
warnings.filterwarnings("ignore")

# from sklearn.linear_model import LinearRegression
# from numpy.random import RandomState
# from sklearn.decomposition import FastICA
# from sklearn.manifold import TSNE
# from sklearn.manifold import Isomap

def load_images_from_folder(folder):
    images = []
    targets = []
    image_names = []
    target = 0  # начальный номер класса

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert('L')  # в оттенки серого
                    img = img.resize((64, 64))  # приводим к размеру 64x64
                    img_array = np.array(img)
                    image_names.append(file)
                    images.append(img_array.reshape(-1))  # <- Ключевое изменение: преобразуем в (4096,)
                    targets.append(target)
                except Exception as e:
                    print(f"Ошибка при загрузке {img_path}: {e}")

        # Увеличиваем target при переходе в новую подпапку (если структура как в Olivetti)
        if dirs:
            target += 1

    if not images:
        raise ValueError(f"В папке {folder} не найдены изображения")

    return np.array(images), np.array(targets), np.array(image_names)

def plot_embedding(X, y, images_small=None, title=None):
    """
    Nice plot on first two components of embedding with Offsets.

    """
    # take only first two columns
    X = X[:, :2]
    # scaling
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(13,8))
    ax = plt.subplot(111)

    for i in range(X.shape[0] - 1):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.RdGy(y[i]),
                 fontdict={'weight': 'bold', 'size': 12})
        if images_small is not None:
            imagebox = OffsetImage(images_small[i], zoom=.4, cmap = 'gray')
            ab = AnnotationBbox(imagebox, (X[i, 0], X[i, 1]),
                xycoords='data')
            ax.add_artist(ab)

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-1:
                # don't show points that are too close
                continue
    if title is not None:
        plt.title(title)

def intrinsic_dim_sample_wise(X, k=5):
    neighb = NearestNeighbors(n_neighbors=k+1).fit(X)
    dist, ind = neighb.kneighbors(X) # distances between the samples and points
    dist = dist[:, 1:] # the distance between the first points to first points (as basis ) equals zero
    # the first non trivial point
    dist = dist[:, 0:k]# including points k-1
    assert dist.shape == (X.shape[0], k) # requirments are there is no equal points
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k-1]) # dinstance betveen the bayeasan statistics
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    return intdim_sample

def intrinsic_dim_scale_interval(X, k1=10, k2=20):
    X = pd.DataFrame(X).drop_duplicates().values # remove duplicates in case you use bootstrapping
    intdim_k = []
    for k in range(k1, k2 + 1): # in order to reduse the noise by eliminating of the nearest neibours
        m = intrinsic_dim_sample_wise(X, k).mean()
        intdim_k.append(m)
    return intdim_k

def repeated(func, X, nb_iter=100, random_state=None, mode='bootstrap', **func_kw):
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    results = []

    iters = range(nb_iter)
    for i in iters:
        if mode == 'bootstrap':# and each point we want to resample with repeating points to reduse the errors
            #232 111 133
            Xr = X[rng.randint(0, nb_examples, size=nb_examples)]
        elif mode == 'shuffle':
            ind = np.arange(nb_examples)
            rng.shuffle(ind)
            Xr = X[ind]
        elif mode == 'same':
            Xr = X
        else:
            raise ValueError('unknown mode : {}'.format(mode))
        results.append(func(Xr, **func_kw))
    return results

def make_collage(folder_path):
    images, targets, image_names = load_images_from_folder(folder_path)

    X = images.reshape(len(images), -1)

    data = images
    target = targets

    U, sigma, V = np.linalg.svd(data)
    sample_size, sample_dim = np.shape(data)

    pca = PCA().fit(data)

    X_projected = PCA(136).fit_transform(data)
    data_pic = data.reshape((-1, 64, 64))

    k1 = 1 # start of interval(included)
    k2 = 5 # end of interval(included)
    nb_iter = 3 # more iterations more accuracy
    intdim_k_repeated = repeated(intrinsic_dim_scale_interval, # intrinsic_dim_scale_interval gives better estimation
                                 data,
                                 mode='bootstrap',
                                 nb_iter=nb_iter, # nb_iter for bootstrapping
                                 k1=k1, k2=k2)
    intdim_k_repeated = np.array(intdim_k_repeated)

    x = np.arange(k1, k2+1)

    # let's leave 20 peaple from faces to get more comprehencible visualisation

    data = images
    target = targets

    data = data[target <20]
    target = target[target <20]

    # X_projected = FastICA(20, random_state = 42).fit_transform(data)
    # data_pic = data.reshape((-1, 64, 64))
    #
    # tsne = TSNE(n_components=2, n_iter = 1000, metric='euclidean', learning_rate= 10, verbose=2, random_state = 42)
    #
    # X_projected = tsne.fit_transform(data)
    # data_pic = data.reshape((-1, 64, 64))
    #
    # X_projected = Isomap(n_components=2).fit_transform(data)
    # data_pic = data.reshape((-1, 64, 64))

    X_projected = MDS(n_components=2).fit_transform(data)
    data_pic = data.reshape((-1, 64, 64))

    # Исходный массив
    arr = X_projected

    # 1. Сортируем массив по X (первый столбец)
    sorted_by_x = arr[arr[:, 0].argsort()]

    error_rate = 10

    # 2. Выбираем левую, центральную и правую точки
    left_point = sorted_by_x[0]      # Минимальный X
    right_point = sorted_by_x[-1]    # Максимальный X
    center_point = sorted_by_x[len(sorted_by_x) // 2]  # Медиана по X

    print("Левая точка (min X):", left_point)
    print("Центральная точка (медиана X):", center_point)
    print("Правая точка (max X):", right_point)

    diff_images = []
    for i in range(len(X_projected)):
      if (X_projected[i][0] == left_point[0] and X_projected[i][1] == left_point[1]) \
      or (X_projected[i][0] == center_point[0] and X_projected[i][1] == center_point[1]) \
      or (X_projected[i][0] == right_point[0] and X_projected[i][1] == right_point[1]):
        diff_images.append(i)
        print(f'{i} - {X_projected[i]}')

    files_for_concat = []

    for num in diff_images:
      file_name = os.path.join(folder_path, image_names[num])
      # img = mpimg.imread(file_name)
      files_for_concat.append(file_name)
      # plt.imshow(img)
      # plt.show()

    # Загружаем изображения
    img1 = Image.open(files_for_concat[0])
    img2 = Image.open(files_for_concat[1])
    img3 = Image.open(files_for_concat[2])

    # Проверяем, что все изображения имеют одинаковую высоту
    assert img1.height == img2.height == img3.height, "Изображения должны быть одинаковой высоты"

    # Создаем новое изображение
    total_width = img1.width + img2.width + img3.width
    new_img = Image.new('RGB', (total_width, img1.height))

    # Вставляем изображения
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    new_img.paste(img3, (img1.width + img2.width, 0))

    num = folder_path.split(sep='/')[-1].split(sep='_')[0]
    new_img.save(f"./example_material/collages/{num}.jpg")


    # folder_path = "./example_material/rendered_imgs/00000002_1ffb81a71e5b402e966b9341_trimesh_001"

def create_photo_collage():
    object_path = "C:\\Users\\mminecz\\PycharmProjects\\PythonProject\\3dLLM_Lambda\\example_material\\rendered_imgs"

    # Проходим по всем подпапкам и файлам
    for dirpath, dirnames, filenames in os.walk(object_path):

        for dirname in dirnames:
            path = os.path.join('./example_material/rendered_imgs/', dirname)
            make_collage(path)
        # for filename in filenames:
        #     if filename.endswith(".obj"):  # можно убрать, если нужны все файлы
        #         file_path = os.path.join(dirpath, filename)
        #         part_number = dirpath.split(sep='\\')[-1]
        #
        #         if part_number in parts:
        #             obj_files.append(os.path.join(f"./abc_dataset/object/{part_number}/", filename))

                # with open(file_path, 'r', encoding='utf-8') as file:
                #     try:
                #         data = yaml.safe_load(file)
                #         if part_number in parts:
                #             obj_files.append(part_number)
                #     except Exception as e:
                #         print(f"Ошибка при чтении {file_path}: {e}")

create_photo_collage()