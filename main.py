from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Гистограмма
def to_hist(arr):
    X, Y = arr.shape
    result = np.zeros(256)
    for i in range(X):
        for j in range(Y):
            result[arr[i, j]] += 1
    return result


# ЧБ
def black_and_white(arr):
    X, Y, U = arr.shape
    result = np.zeros((X, Y), dtype=int)
    for i in range(X):
        for j in range(Y):
            result[i, j] = np.mean(arr[i, j])
    return np.array(result, dtype=np.uint8)


def otsu_binarization(arr):
    result = black_and_white(arr)
    X, Y = result.shape
    max_sigma = 0
    max_t = 0

    for t in range(1, 255):
        class0 = result[np.where(result < t)]
        mean0 = np.mean(class0) if len(class0) > 0 else 0
        weight0 = len(class0) / (X * Y)
        class1 = result[np.where(result >= t)]
        mean1 = np.mean(class1) if len(class1) > 0 else 0
        weight1 = len(class1) / (X * Y)
        sigma = weight0 * weight1 * ((mean0 - mean1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = t

    result[result < max_t] = 0
    result[result >= max_t] = 255
    return result, max_t


def salt_and_pepper(arr):
    X, Y = arr.shape
    borderX, borderY = arr.shape
    # Добавляем рамку
    borderX += 2
    borderY += 2
    result = np.zeros((borderX, borderY), dtype=np.uint8)
    # Внутрь помещаем изображение
    result[1:-1, 1:-1] = arr
    salt = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]], dtype=np.uint8)
    pepper = np.array([[255, 255, 255], [255, 0, 255], [255, 255, 255]], dtype=np.uint8)
    countSalt = 0
    countPepper = 0
    for i in range(X):
        for j in range(Y):
            if np.array_equal(result[i - 1:i + 2, j - 1:j + 2], salt):
                result[i, j] = 0
                countSalt += 1
            elif np.array_equal(result[i - 1:i + 2, j - 1:j + 2], pepper):
                result[i, j] = 1
                countPepper += 1
    print(f"Соли: {countSalt}\nПерца: {countPepper}\n")
    return result[1:-1, 1:-1]


def seeds_growing(arr):
    X, Y = arr.shape
    result = np.zeros((X, Y, 3), dtype=np.uint8)
    directions = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    marked = np.zeros((X, Y), dtype=np.uint8)
    seeds = []
    # Проходим по всем пикселям, генерируем цвет
    for i in range(len(marked)):
        for j in range(len(marked[0])):
            if marked[i, j] == 1:
                continue
            seeds.append((i, j))
            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            # Проходим по связанной с данным пикселем области
            while len(seeds):
                seed = seeds.pop()
                x, y = seed[0], seed[1]
                # Помечаем текущий пиксель
                marked[x, y] = 1
                result[x, y] = color
                # Расходимся по направлениям
                for direction in directions:
                    next_x = x + direction[0]
                    next_y = y + direction[1]
                    # Проверяем, что мы не вышли за пределы картинки
                    if 0 < next_x < X and 0 < next_y < Y:
                        if (not marked[next_x, next_y]) and (arr[next_x, next_y] == arr[x, y]):
                            result[next_x, next_y] = color
                            marked[next_x, next_y] = 1
                            seeds.append((next_x, next_y))

    return result


def hist_method(arr, near):
    result = black_and_white(arr)
    old_hist = to_hist(result)

    local_mins = []

    plt.subplot(2, 1, 1)
    plt.bar(range(256), old_hist)

    temp = 0
    for i in range(256):
        cur_bin = old_hist[i]
        neighbour = old_hist[max(0, i - near): min(i + near + 1, 256)]
        if np.all(cur_bin <= neighbour):
            if len(local_mins) != 0:
                if i - temp >= near:
                    local_mins = np.append(local_mins, i)
                    temp = i
            else:
                local_mins = np.append(local_mins, i)
                temp = i

    new_result = result.copy()
    for i in range(len(new_result)):
        for j in range(len(new_result[i])):
            for t in range(len(local_mins) - 1):
                if local_mins[t] <= result[i][j] < local_mins[t + 1]:
                    new_result[i][j] = local_mins[t]

    plt.subplot(2, 1, 2)
    plt.bar(range(256), to_hist(new_result))
    plt.show()

    return new_result


# Считываем изображения
image = np.array(Image.open("img_for_report.png"))
plt.subplot(1, 3, 1)
plt.imshow(image)
"""
# 1
imageAngel = np.array(Image.open("one.jpg"))
plt.subplot(3, 3, 1)
plt.imshow(imageAngel)
# 2
imageCat = np.array(Image.open("two.jpg"))
plt.subplot(3, 3, 2)
plt.imshow(imageCat)
# 3
imageDog = np.array(Image.open("three.jpg"))
plt.subplot(3, 3, 3)
plt.imshow(imageDog)
"""

# Метод Оцу
image1, threshold = otsu_binarization(image)
plt.subplot(1, 3, 2)
plt.ylabel(f"Порог: {threshold}")
plt.imshow(image1, cmap='gray')
"""
# 1
imageAngel1, threshold = otsu_binarization(imageAngel)
plt.subplot(3, 3, 4)
plt.ylabel(f"Порог: {threshold}")
plt.imshow(imageAngel1, cmap='gray')
# 2
imageCat1, threshold = otsu_binarization(imageCat)
plt.subplot(3, 3, 5)
plt.ylabel(f"Порог: {threshold}")
plt.imshow(imageCat1, cmap='gray')
# 3
imageDog1, threshold = otsu_binarization(imageDog)
plt.subplot(3, 3, 6)
plt.ylabel(f"Порог: {threshold}")
plt.imshow(imageDog1, cmap='gray')
"""

# Соль и перец
image1 = salt_and_pepper(image1)
"""
imageAngel1 = salt_and_pepper(imageAngel1)
imageCat1 = salt_and_pepper(imageCat1)
imageDog1 = salt_and_pepper(imageDog1)
"""

# Выращивание семян
image1 = seeds_growing(image1)
plt.subplot(1, 3, 3)
plt.imshow(image1)
"""
imageAngel1 = seeds_growing(imageAngel1)
plt.subplot(3, 3, 7)
plt.imshow(imageAngel1)
imageCat1 = seeds_growing(imageCat1)
plt.subplot(3, 3, 8)
plt.imshow(imageCat1)
imageDog1 = seeds_growing(imageDog1)
plt.subplot(3, 3, 9)
plt.imshow(imageDog1)
"""

plt.show()

# Комбинация гистограммного метода и алгоритма «выращивания семян» для сегментации полутонового изображения
image = hist_method(image, 10)
image = seeds_growing(image)
plt.imshow(image)
plt.show()
"""
imageAngel = hist_method(imageAngel, 10)
imageCat = hist_method(imageCat, 10)
imageDog = hist_method(imageDog, 10)

imageAngel = seeds_growing(imageAngel)
plt.subplot(1, 3, 1)
plt.imshow(imageAngel)
imageCat = seeds_growing(imageCat)
plt.subplot(1, 3, 2)
plt.imshow(imageCat)
imageDog = seeds_growing(imageDog)
plt.subplot(1, 3, 3)
plt.imshow(imageDog)
"""

plt.show()


