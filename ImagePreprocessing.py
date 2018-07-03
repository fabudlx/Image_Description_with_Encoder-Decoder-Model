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


def create_captions_and_image_vectors(data = '../Image2SequenceFiles/data', save_loc =  '../Image2SequenceFiles/dictionaries', dataset='train'):
    model = get_vgg16_model()
    #captions
    id_to_caption_dict = get_id_to_caption_dict(data+'/annotations/captions_'+dataset+'2017.json')
    save_dict(id_to_caption_dict, save_loc+'/id_to_caption_'+dataset+'.dict')
    #image vectors
    id_to_path_to_image_dict = get_image_paths_and_names(data+'/'+dataset+'2017')
    id_to_image_vector_dict = get_id_to_vector_dict(id_to_path_to_image_dict, model)
    save_dict(id_to_image_vector_dict, save_loc+'/id_to_vector_'+dataset+'.dict')

    return id_to_image_vector_dict, id_to_caption_dict

