import os

import numpy as np
import tqdm

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans


model = VGG16(weights='imagenet', include_top=False)

vgg16_feature_list = {}

for kind in ['with', 'without']:
    vgg16_feature_list[kind] = []
    for fname in tqdm.tqdm(os.listdir('training/%s' % kind)):
        img = image.load_img('training/%s/%s' % (kind, fname),
            target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        vgg16_feature = model.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list[kind].append(vgg16_feature_np.flatten())
        
vgg16_feature_list_np = np.array(
    vgg16_feature_list['with'] + vgg16_feature_list['without'])

labels = KMeans(n_clusters=2, random_state=0).fit_predict(vgg16_feature_list_np)
with_labels = labels[:len(vgg16_feature_list['with'])]
without_labels = labels[len(with_labels):]

# find the most common label for the first set, assume it's right
(_, idx, counts) = np.unique(with_labels, return_index=True, return_counts=True)
index = idx[np.argmax(counts)]
with_label = labels[index]
without_label = 1 - with_label

# compute how many are correct
print(100 * len(with_labels[with_labels == with_label]) / len(with_labels),
    '% of the "with" set labeled correctly"')
print(100 * len(without_labels[without_labels == without_label]) / len(without_labels),
    '% of the "without" set labeled correctly"')