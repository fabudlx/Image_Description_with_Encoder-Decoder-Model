from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle
from collections import defaultdict
import json


def get_vgg16_model():
    return VGG16(weights='imagenet', include_top=False, pooling='max')

# def get_vgg16_model():
#     vgg = VGG16(weights='imagenet', include_top=True)
#     vgg.layers.pop()
#     vgg.outputs = [vgg.layers[-1].output]
#     vgg.layers[-1].outbound_nodes = []
#     vgg.summary()
#     return vgg


def get_image_paths_and_names(path):
    all_images_full_path = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    all_images_name = [f for f in listdir(path) if isfile(join(path, f))]
    all_image_ids = [int(image_name.replace('.jpg', '')) for image_name in all_images_name]
    return dict(zip(all_image_ids, all_images_full_path))


def get_id_to_vector_dict(id_to_path_dict, model):
    id_to_vector = {}
    for id, path in id_to_path_dict.items():
        id_to_vector[id] = predict(path, model)[0]
    return id_to_vector


def save_dict(dict, path):
    with open(path, 'wb') as dict_file:
        pickle.dump(dict, dict_file)


def predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)


def get_id_to_caption_dict(path):
    with open(path) as json_file:
        captions_json = json.loads(json_file.read())
        id_to_caption = defaultdict(list)
    for caption in captions_json['annotations']:
        id_to_caption[caption['image_id']].append(caption['caption'])
    return id_to_caption


model = get_vgg16_model()
#validation data
id_to_path_dict = get_image_paths_and_names(r'./files/data/val2017')
id_to_vector = get_id_to_vector_dict(id_to_path_dict, model)
save_dict(id_to_vector, './save_model/id_to_vector_val.dict')
id_to_caption = get_id_to_caption_dict(r'./files/data/annotations/captions_val2017.json')
save_dict(id_to_caption, './save_model/id_to_caption_val.dict')

# id_to_path_dict = get_image_paths_and_names(r'.\files\data\train2017')
# id_to_vector = get_id_to_vector_dict(id_to_path_dict, model)
# save_dict(id_to_vector, './save_model/id_to_vector_train.dict')
# id_to_caption = get_id_to_caption_dict(r'.\files\data\annotations\captions_train2017.json')
# save_dict(id_to_caption, './save_model/id_to_caption_train.dict')

