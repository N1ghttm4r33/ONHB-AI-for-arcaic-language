import tensorflow as tf
import tensorflow_datasets as tfds
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from google.colab import drive

import cv2

import idx2numpy

import struct
import numpy as np

import pytesseract
from PIL import Image
import re
from deslant_img import deslant_img
import concurrent.futures

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate

import tensorflow_model_optimization as tfmot

try:
    # Verifica se há uma TPU disponível
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Executando na TPU:", tpu.master())
except ValueError:
    try:
        # Tenta usar uma GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            strategy = tf.distribute.MirroredStrategy(devices=gpus)
            print("Executando na GPU")
        else:
            # Se não houver TPU nem GPU, usa a CPU
            strategy = tf.distribute.get_strategy()
            print("Executando na CPU")
    except:
        # Se ocorrer algum erro, também usa a CPU
        strategy = tf.distribute.get_strategy()
        print("Executando na CPU")

print("Número de aceleradores:", strategy.num_replicas_in_sync)

drive.mount('/content/drive')

def load_emnist():
    # Caminhos dos arquivos
    train_images_path = "/content/drive/MyDrive/dataset/train/emnist-byclass-train-images-idx3-ubyte"
    train_labels_path = "/content/drive/MyDrive/dataset/train/emnist-byclass-train-labels-idx1-ubyte"
    test_images_path = "/content/drive/My Drive/dataset/test/emnist-byclass-test-images-idx3-ubyte"
    test_labels_path = "/content/drive/MyDrive/dataset/test/emnist-byclass-test-labels-idx1-ubyte"
    # Carregar os dados de treinamento
    data_train = idx2numpy.convert_from_file(train_images_path)
    train_label = idx2numpy.convert_from_file(train_labels_path)

    # Carregar os dados de teste
    data_test = idx2numpy.convert_from_file(test_images_path)
    test_label = idx2numpy.convert_from_file(test_labels_path)

    # Converter os arrays NumPy em objetos tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((data_train, train_label))
    test_dataset = tf.data.Dataset.from_tensor_slices((data_test, test_label))

    return train_dataset, test_dataset

def illumination_compensation(image):
    # Definir o kernel gaussiano
    gaussian_kernel = tf.constant([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=tf.float32) / 16.0
    gaussian_kernel = tf.reshape(gaussian_kernel, [3, 3, 1, 1])

    # Expandir as dimensões da imagem para incluir a dimensão do batch
    image = tf.expand_dims(image, axis=0)
    image = tf.expand_dims(image, axis=-1)
    image = tf.cast(image, tf.float32)

    # Aplicar um filtro gaussiano para suavizar a imagem usando TensorFlow
    background = tf.nn.depthwise_conv2d(image, gaussian_kernel, strides=[1, 1, 1, 1], padding='SAME')

    # Subtrair a imagem suavizada da imagem original
    corrected_image = tf.subtract(image, background)

    return tf.squeeze(corrected_image, axis=[0, -1])

def preprocess(image, label):
    image = tf.cast(image, tf.float32)

    image = illumination_compensation(image)

    # Normaliza as imagens
    images = tf.cast(image, tf.float32) / 255.0

    return images, label

def preprocess2(image, label):
    #segunda camada de pré-processamento
    image = illumination_compensation(image)

    return image, label

root_dir = '/content/drive/MyDrive/dataset/archive/data'
save_dir = '/content/drive/MyDrive/dataset/data2'
encoder = LabelEncoder()  # Crie o codificador uma vez
unique_characters = set()

def process_image(image_path):
    # Carregar a imagem
    imagem = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    copy_original_image = imagem

    #corta 300px
    imagem = imagem[300:]
    #deixa apenas os primeiros 420px
    imagem = imagem[:420]

    # Aplicar um filtro de threshold para destacar o texto
    _, imagem_threshold = cv2.threshold(imagem, 150, 255, cv2.THRESH_BINARY)

    # Aplicar um desfoque gaussiano para reduzir o ruído
    imagem_blur = cv2.GaussianBlur(imagem_threshold, (5, 5), 0)

    # Usar pytesseract para extrair o texto
    texto = pytesseract.image_to_string(imagem_blur)

    # Dividir o texto em linhas
    linhas = texto.split('\n')

    # Remover espaços em branco antes e depois de cada linha
    texto = [linha.strip() for linha in linhas]

    # Concatenar as 9 linhas em uma única string
    texto_concatenado = ' '.join(texto)

    for character in texto_concatenado:
        unique_characters.add(character)

    # Pré-processe a imagem
    image = copy_original_image[720:2032]

    image = cv2.resize(imagem, (2447, 1280))
    image = np.array(image) / 255.0

    return image, texto_concatenado

def save_as_ubyte(images, labels, image_file, label_file, save_dir):
    # Converter as imagens e rótulos para o tipo de dados ubyte
    images_ubyte = (images * 255).astype(np.ubyte)
    labels_ubyte = labels.astype(np.ubyte)

    # Salvar as imagens e rótulos como arquivos ubyte
    image_path = os.path.join(save_dir, image_file)
    label_path = os.path.join(save_dir, label_file)

    with open(image_path, 'wb') as f:
        f.write(images_ubyte.tobytes())

    with open(label_path, 'wb') as f:
        f.write(labels_ubyte.tobytes())

def image_dataset():
    # Listas para armazenar as imagens e rótulos
    train_data2_images = []
    train_data2_labels = []

    # Percorra todas as subpastas
    for folder in os.listdir(root_dir):

        folder_path = os.path.join(root_dir, folder)

        # Percorra todas as imagens na pasta
        with concurrent.futures.ProcessPoolExecutor() as executor:
            image_files = [os.path.join(folder_path, image_file) for image_file in os.listdir(folder_path)]
            results = list(executor.map(process_image, image_files))

        for image, label in results:
            # Adicione a imagem e o rótulo às listas
            if image is not None and label is not None:
                train_data2_images.append(image)
                train_data2_labels.append(label)

    # Ajuste o codificador ao conjunto completo de caracteres únicos
    encoder.fit(list(unique_characters))

    # Transforme os rótulos das imagens
    train_data2_labels = [encoder.transform([label])[0] for label in train_data2_labels]

    # Converta as listas para arrays numpy para uso com o modelo
    data2_images = np.array(train_data2_images)
    data2_labels = np.array(train_data2_labels)

    # Converter os rótulos inteiros para codificação one-hot
    label_one_hot = to_categorical(data2_labels, num_classes=len(unique_characters))
    label = np.squeeze(label_one_hot)

    # Salve as imagens e rótulos como arquivos ubyte
    save_as_ubyte(data2_images, label, 'train_images2.ubyte', 'train_labels2.ubyte', save_dir)

    return data2_images, train_data2_labels


def tensor_images():
    # Obtenha os dados
    data2_images, data2_labels = image_dataset()

    # Embaralhe os dados
    indices = tf.range(start=0, limit=tf.shape(data2_images)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    # Divida os índices para os dados de treinamento e teste
    train_fraction = 0.8  # 80% para treinamento, 20% para teste
    train_size = tf.cast(train_fraction * tf.cast(tf.shape(data2_images)[0], tf.float32), tf.int32)
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    # Crie os conjuntos de dados de treinamento e teste
    train_images = tf.gather(data2_images, train_indices)
    train_labels = tf.gather(data2_labels, train_indices)
    test_images = tf.gather(data2_images, test_indices)
    test_labels = tf.gather(data2_labels, test_indices)

    # Crie os conjuntos de dados TensorFlow
    train2_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test2_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    return train2_dataset, test2_dataset

# Carrega os dados e aplica pré-processamento
train_dataset, test_dataset = load_emnist()
BATCH_SIZE = 64 * strategy.num_replicas_in_sync
train_dataset = train_dataset.map(preprocess).shuffle(10000).batch(BATCH_SIZE)
test_dataset = test_dataset.map(preprocess).batch(BATCH_SIZE)

train_dataset2, test_dataset2 = tensor_images()
BATCH_SIZE = 64 * strategy.num_replicas_in_sync
train_dataset2 = train_dataset2.map(preprocess2).shuffle(10000).batch(BATCH_SIZE)
test_dataset2 = test_dataset2.map(preprocess2).batch(BATCH_SIZE)

# Codificar os caracteres como inteiros
encoder = tf.keras.layers.experimental.preprocessing.StringLookup()

def create_model():
    # Defina as camadas compartilhadas
    shared_layers = [
        tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.SeparableConv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Reshape(target_shape=((16, 16, 16*16))),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(512, return_sequences=True)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.1)
    ]

    # Aplique a camada Embedding aos rótulos de texto separadamente
    text_labels = encoder(text_labels))  # Supondo que text_labels são seus rótulos de texto
    text_labels = tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True)(text_labels)

    for layer in shared_layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)

    model = tf.keras.Sequential(shared_layers)

    # Aplicar quantização ao modelo
    quantized_model = tfmot.quantization.keras.quantize_model(model)

    pruned_model = tf.keras.tfmot.sparsity.keras.prune_low_magnitude(quantized_model)

    emnist_output = tf.keras.layers.Dense(10, activation='softmax')(pruned_model.output)

    text_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(62, activation='sigmoid'))(pruned_model.output)
    text_output = tf.keras.layers.LSTM(128, return_sequences=True)(text_output)
    text_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(62, activation='sigmoid'))(text_output)

    combined_model = Model(inputs=pruned_model.input, outputs=[emnist_output, text_output])

    return combined_model

def train_model():
    with strategy.scope():
        # Verifica se o modelo já existe
        if os.path.exists('/content/drive/MyDrive/tst1'):
            print("Carregando o modelo existente para treinamento adicional...")
            model = load_model('/content/drive/MyDrive/tst1')
        else:
            print("Criando um novo modelo para treinamento...")
            model = create_model()
            model.save('/content/drive/MyDrive/tst1')

        # Compilar o modelo
        model.compile(optimizer='adam', loss=['categorical_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])

        # Definir o EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        # Definir o ModelCheckpoint para salvar o modelo a cada validação
        model_checkpoint = ModelCheckpoint('/content/drive/MyDrive/ONHB', save_best_only=True, monitor='val_loss', mode='min')

        # Treinar o modelo com validação a cada 5 épocas
        for i in range(10):
            # Treinar o modelo
            model.fit(x=[train_dataset, train_dataset2], y=[train_dataset, train_dataset2], epochs=10, validation_data=[test_dataset, test_dataset2], callbacks=[early_stopping, model_checkpoint])

        # Salva o modelo treinado na pasta ONHB
        model.save('/content/drive/MyDrive/tst1')

    return model

# Treina o modelo
trained_model = train_model()

