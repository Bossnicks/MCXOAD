import mglearn
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
#PCAнаходит оптимальные направления с точки зрения реконструкции данных
#Анализ главных компонент(PSA)
#Мы  можем  использовать PCAдля  уменьшения  размерности,  сохранив  лишь несколько  главных  компонент.
mglearn.plots.plot_pca_illustration()
plt.show()#Рис1(главные компоненты (показывают основные напрвления дисперсии))

print("--------------------------------Применение PCA к набору данных cancerдля визуализации--------------------------------")
#ВИЗУАЛИЗАЦИЯ ВЫСОКОРАЗМЕРНЫХ НАБОРОВ ДАННЫХ(постороение диагрмм рассеивания )

#вычислив гистограммы распределения значений признаков для двух классов, доброкачественных и злокачественных опухолей
cancer = load_breast_cancer()
fig, axes = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]
ax = axes.ravel()


for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())

ax[0].set_xlabel("Значение признака")
ax[0].set_ylabel("Частота")
ax[0].legend(["доброкачественная", "злокачественная"], loc="best")
fig.tight_layout()
plt.show()

# Подготовка данных для PCA
print("------------------------Тот же набор но уже  с использованием PCA------------------------")
#тмасштабируем наши данные таким образом, чтобы каждый признак имел единичную дисперсию,
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

# а затем применяем вращение  и снижение размерности, вызвав метод transform.
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
#По умолчанию PCAлишь поворачивает (и  смещает)  данные,  но  сохраняет  все  главные № компоненты.
#указ количество компонент

print("Форма исходного массива:{}".format(str(X_scaled.shape)))
print("Форма массива после сокращения размерности:{}".format(str(X_pca.shape)))

# Визуализация PCA
print("----------------График первых двух главных компонент----------------")
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")
plt.show()
#разделение классов прошло хорошо и даже если использовать линейный классификатоп то все получится
#так как метод бзе учителя просто анализирует корреляционные связи в данных.

#!!!!НЕ PCA :две оси графика часто бывает сложно интерпретировать(главные компоненты соответсвуют направлениям данных
# (предсталвют собой комбинации исхождных признаков (СВАМИ ПО СЕБЕ ОЧЕНЬ СЛОЖНЫ )))

print("Форма главных компонент:{}".format(pca.components_.shape))#содержаться сами главные компоненты в атрибуте pca.components
print("Компоненты PCA:\n{}".format(pca.components_))

plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["Первая компонента", "Вторая компонента"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel("Характеристика")
plt.ylabel("Главные компоненты")
plt.show()#Каждая строка в атрибуте components_соответствует одной главной компоненте и они отсортированы по важности

# Данные LFW
print("-------------------------------------Метод «Собственых лиц»(eigenfaces) для выделения характеристик-------------------------------------")
#!!!!! одно из ПРИМЕНЕНИЕ PCA : выделение признаков( заключается в поиске нового представления данных,  которое  в  отличие от  исходного лучше подходит для  анализа)
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)#набор содержит лица знаменитостей
image_shape = people.images[0].shape

fix, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})

for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])

plt.show()
print("Форма массива изображений лиц:{}".format(people.images.shape))
print("Количество классов:{}".format(len(people.target_names)))

# KNeighborsClassifier
#вычисляем частоту встречаемости каждого ответа
counts = np.bincount(people.target)
#печатаем частоты рядом с ответами
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25}{1:3}".format(name, count), end='')
    if (i + 1) % 3 == 0:
        print()#Чтобы данные стали менее асимметричными, мы будем рассматривать не более 50  изображений  каждого  человек

mask = np.zeros(people.target.shape, dtype=np.bool)

for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.
#Общая  задача  распознавания  лиц  заключается  в  том,  чтобы  спросить,  не принадлежит  ли  незнакомое  фото  уже  известному  человеку  из  базы  данных.
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
print("Используем классификатор ближайшего дял решения задачи ")
#классификатор
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Правильность на тестовом наборе для 1-nn:{:.2f}".format(knn.score(X_test, y_test)))

#И  вот  именно  здесь  применяется PCA.(Неудачно) Используя пиксельное представление для сопоставления двух изображений

# Здесь  мы воспользуемся опцией PCA выбеливание (whitening)
#(которая преобразует  компоненты  к    одному    и    тому    же    масштабу(
mglearn.plots.plot_pca_whitening()
plt.show()

pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("Обучающие данные после PCA:{}".format(X_train_pca.shape))
#Новые данные содержат 100 новых признаков, первые 100 главных компонент(использовать новое предсталвение ,чтобы классифицировать изображения)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("Правильность на тестовом наборе:{:.2f}".format(knn.score(X_test_pca, y_test)))

print("Форма pca.components_:{}".format(pca.components_.shape))

mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)
plt.show()

mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")
plt.show()




#Похоже, что первая компонента главным образом кодирует контраст  между  лицом  и  фоном
#вторая  компонента  кодирует  различия  в освещенности между  правой  и  левой половинами лица  и  т

#ВЫВОД:Мы кратко рассказали  о  преобразовании PCAкак  способе  поворота  данных  с  последующим удалением компонент, имеющих низкую дисперсию
#Еще одна полезная интерпретация заключается в том, чтобы попытаться вычислить значения новых признаков, полученные после  поворота PCA,  таким  образом,
# мы  можем  записать  тестовые  точки  в  виде взвешенной суммы главных компонен

#!!!!!!!!!!!!!!!!!модель PCA–реконструировать  исходные  данные,  используя  лишь  некоторые компоненты.