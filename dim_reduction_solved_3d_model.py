import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
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

def load_images_from_folder(folder):
    images = []
    targets = []
    image_names = []
    target = 0  # начальный номер класса

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('_r.png', '_g.png', '_b.png')):
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

def get_farthest_points(points, k):
    """Выбирает k самых удалённых точек из массива."""
    n = points.shape[0]
    selected_indices = []

    # 1. Начинаем с самой удалённой точки от центра масс
    centroid = np.mean(points, axis=0)
    distances = cdist(points, centroid[np.newaxis, :])
    first_idx = np.argmax(distances)
    selected_indices.append(first_idx)

    # 2. Последовательно добавляем самые удалённые от уже выбранных
    while len(selected_indices) < k:
        # Вычисляем минимальное расстояние от каждой точки до уже выбранных
        min_distances = np.min(cdist(points, points[selected_indices]), axis=1)
        # Выбираем точку с максимальным минимальным расстоянием
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)

    return points[selected_indices]

def make_collage(folder_path, part_number):
    images, targets, image_names = load_images_from_folder(folder_path)

    X = images.reshape(len(images), -1)

    data = images
    target = targets

    U, sigma, V = np.linalg.svd(data)
    sample_size, sample_dim = np.shape(data)

    pca = PCA().fit(data)

    X_projected = PCA(60).fit_transform(data)
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

    X_projected = Isomap(n_components=2).fit_transform(data)
    data_pic = data.reshape((-1, 64, 64))

    create_photo_collage(folder_path, X_projected, image_names, part_number)

def create_photo_collage(folder_path, X_projected, image_names, part_number):
    for count in [3, 4, 6]:
        farthest_points = get_farthest_points(X_projected, count)

        diff_images = []
        for i in range(len(X_projected)):
            for j in range(len(farthest_points)):
                if (X_projected[i][0] == farthest_points[j][0] and X_projected[i][1] == farthest_points[j][1]):
                    print(f'{i} - {X_projected[i]} and {j} - {farthest_points[j]}')
                    diff_images.append(i)

        files_for_concat = []

        for num in diff_images:
            file_name = os.path.join(folder_path, image_names[num])
            img = mpimg.imread(file_name)
            files_for_concat.append(file_name)

        images = []
        for i in range(len(files_for_concat)):
            img = mpimg.imread(files_for_concat[i])
            img = Image.fromarray((img * 255).astype(np.uint8))
            images.append(img)

        save_collage(count, images, part_number)

def save_collage(count, images, part_number):
    if count == 3:
        # Создаем новое изображение
        total_width = images[0].width * 3
        total_height = images[0].height
        new_img = Image.new('RGB', (total_width, total_height))

        # Вставляем изображения
        new_img.paste(images[0], (0, 0))
        new_img.paste(images[1], (images[0].width, 0))
        new_img.paste(images[2], (images[0].width + images[1].width, 0))

        new_img.save(f"./example_material/collages_3/{part_number}.jpg")

    if count == 4:
        # Создаем новое изображение
        total_width = images[0].width * 2
        total_height = images[0].height * 2
        new_img = Image.new('RGB', (total_width, total_height))

        # Вставляем изображения
        new_img.paste(images[0], (0, 0))
        new_img.paste(images[1], (images[0].width, 0))
        new_img.paste(images[2], (0, images[0].height))
        new_img.paste(images[3], (images[0].width, images[0].height))

        new_img.save(f"./example_material/collages_4/{part_number}.jpg")

    if count == 6:
        # Создаем новое изображение
        total_width = images[0].width * 3
        total_height = images[0].height * 2
        new_img = Image.new('RGB', (total_width, total_height))

        # Вставляем изображения
        new_img.paste(images[0], (0, 0))
        new_img.paste(images[1], (images[0].width, 0))
        new_img.paste(images[2], (images[0].width + images[1].width, 0))
        new_img.paste(images[3], (0, images[0].height))
        new_img.paste(images[4], (images[3].width, images[0].height))
        new_img.paste(images[5], (images[3].width + images[4].width, images[0].height))

        new_img.save(f"./example_material/collages_6/{part_number}.jpg")

def main():
    # object_path = "C:\\Users\\mminecz\\PycharmProjects\\main\\3D_LLM\\DiffuRank\\example_material\\rendered_imgs"
    # folder_path = "./example_material/rendered_imgs/00000002_1ffb81a71e5b402e966b9341_trimesh_001"
    object_path = "./example_material/rendered_imgs"

    # Проходим по всем подпапкам и файлам
    for dirpath, dirnames, filenames in os.walk(object_path):
        if dirpath != object_path:
            part_number = dirpath.split(sep='\\')[-1].split(sep='_')[0]
            print(dirpath)
            make_collage(dirpath, part_number)

main()