import numpy as np
import pandas as pd
import pydicom
import glob
import random
from skimage.transform import resize


def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


def generate_dataset(glob_pattern, labels, input_size=1024, output_size=256,
                     num_channels=1, validation_percentage=0.1):

    def helper(images):

        X_resized = np.zeros(
            (len(images), output_size, output_size, num_channels), dtype=np.float32)
        Y_resized = np.zeros(
            (len(images), output_size, output_size, 1), dtype=np.bool)

        X = np.zeros((len(images), input_size, input_size,
                      num_channels), dtype=np.uint8)
        Y = np.zeros((len(images), input_size, input_size, 1), dtype=np.bool)

        df_full = pd.read_csv(labels, index_col='ImageId')

        for n, image in enumerate(images):
            dataset = pydicom.read_file(image)
            X[n] = np.expand_dims(dataset.pixel_array, axis=2)
            try:
                if '-1' in df_full.loc[image.split('/')[-1][:-4], ' EncodedPixels']:
                    Y[n] = np.zeros((input_size, input_size, 1))
                else:
                    if type(df_full.loc[image.split('/')[-1][:-4], ' EncodedPixels']) == str:
                        Y[n] = np.expand_dims(
                            rle2mask(df_full.loc[image.split('/')[-1][:-4], ' EncodedPixels'],
                                     input_size, input_size).T, axis=2)
                    else:
                        Y[n] = np.zeros((input_size, input_size, 1))
                        for x in df_full.loc[image.split('/')[-1][:-4], ' EncodedPixels']:
                            Y[n] = Y[n] + \
                                np.expand_dims(
                                    rle2mask(x, input_size, input_size).T, axis=2)
            except KeyError:
                print(
                    f"Key {image.split('/')[-1][:-4]} without mask, assuming healthy patient.")
                # Assume missing masks are empty masks.
                Y[n] = np.zeros((input_size, input_size, 1))

            X_resized[n] = resize(X[n], (output_size, output_size, 1))
            Y_resized[n] = resize(Y[n], (output_size, output_size, 1))

        return X_resized, Y_resized

    data_files = glob.glob(glob_pattern)
    dataset_size = len(data_files)
    valid_size = int(dataset_size*validation_percentage)
    train_size = dataset_size - valid_size
    print(f'dataset size           : {dataset_size} dicom files')
    print(f'training dataset size  : {train_size}  dicom files')
    print(f'validation dataset size: {valid_size}   dicom files')

    random.shuffle(data_files)
    train_files = data_files[:train_size]
    valid_files = data_files[train_size:]

    print('generating training dataset')
    X_train, Y_train = helper(train_files)

    print('generating validation dataset')
    X_valid, Y_valid = helper(valid_files)

    return X_train, Y_train, X_valid, Y_valid


if __name__ == "__main__":
    x, y = generate_dataset('/Users/upul/Experiments/Artificial-Intelligence-for-Medicine/data/pneumothorax_segmentation/dicom-images-train/*/*/*.dcm',
                            '/Users/upul/Experiments/Artificial-Intelligence-for-Medicine/data/pneumothorax_segmentation/train-rle.csv')
