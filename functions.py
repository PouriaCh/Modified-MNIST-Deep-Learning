# Functions
import numpy as np

def recur(i, j, train_images1, detected):
    detected.append([i, j])
    if i + 1 < 63 and [i + 1, j] not in detected:
        if train_images1[i + 1][j] == 1:
            recur(i + 1, j, train_images1, detected)
    if j + 1 < 63 and [i, j + 1] not in detected:
        if train_images1[i][j + 1] == 1:
            recur(i, j + 1, train_images1, detected)
    if i - 1 > 0 and [i - 1, j] not in detected:
        if train_images1[i - 1][j] == 1:
            recur(i - 1, j, train_images1, detected)
    if j - 1 > 0 and [i, j - 1] not in detected:
        if train_images1[i][j - 1] == 1:
            recur(i, j - 1, train_images1, detected)


def preprocess(input_images):
    croped = []
    croped_image = []
    for img_index in range(len(input_images)):
        print("processing image " + str(img_index) + " of " + str(len(input_images)), end='\r')
        input_images1 = input_images[img_index] > 250

        obj = []
        obj_number = -1
        for i in range(64):
            for j in range(64):
                detected = []
                if obj_number == -1:
                    if input_images1[i][j] == 1:
                        recur(i, j, input_images1, detected)
                        down = max([_[0] for _ in detected])
                        up = min([_[0] for _ in detected])
                        right = max([_[1] for _ in detected])
                        left = min([_[1] for _ in detected])

                        if down - up + 1 > right - left + 1:
                            area = (down - up + 1) ** 2
                        elif down - up + 1 <= right - left + 1:
                            area = (right - left + 1) ** 2
                        obj.append([up, down, left, right, area])
                        obj_number += 1

                elif input_images1[i][j] == 1 and not (
                        i in range(obj[obj_number][0], obj[obj_number][1] + 1) and j in range(obj[obj_number][2],
                                                                                              obj[obj_number][3] + 1)):
                    recur(i, j, input_images1, detected)
                    down = max([_[0] for _ in detected])
                    up = min([_[0] for _ in detected])
                    right = max([_[1] for _ in detected])
                    left = min([_[1] for _ in detected])

                    if down - up + 1 > right - left + 1:
                        area = (down - up + 1) ** 2
                    elif down - up + 1 <= right - left + 1:
                        area = (right - left + 1) ** 2
                    obj.append([up, down, left, right, area])
                    obj_number += 1

        maximum = -1
        for i in range(len(obj)):

            if obj[i][4] > maximum:
                maximum = obj[i][4]
                index = i

        croped.append(obj[index])

        croped_image.append(input_images1[obj[index][0]:obj[index][1] + 1, obj[index][2]:obj[index][3] + 1])

        while np.shape(croped_image[img_index])[0] < 64:
            croped_image[img_index] = np.insert(croped_image[img_index], 0,
                                                np.zeros(np.shape(croped_image[img_index])[1]), axis=0)
            if np.shape(croped_image[img_index])[0] < 64:
                croped_image[img_index] = np.insert(croped_image[img_index], np.shape(croped_image[img_index])[0],
                                                    np.zeros(np.shape(croped_image[img_index])[1]), axis=0)

        while np.shape(croped_image[img_index])[1] < 64:
            croped_image[img_index] = np.insert(croped_image[img_index], 0,
                                                np.zeros(np.shape(croped_image[img_index])[0]), axis=1)
            if np.shape(croped_image[img_index])[1] < 64:
                croped_image[img_index] = np.insert(croped_image[img_index], np.shape(croped_image[img_index])[1],
                                                    np.zeros(np.shape(croped_image[img_index])[0]), axis=1)

    croped_image = np.asarray(croped_image)

    return croped_image

# Function to write test predictions to .csv file

def csvWriter(prediction, submission_no):
    index = 0
    filename = 'G_40_submission_' + str(submission_no) + '.csv'
    csv = open(filename, "w")

    columnTitleRow = "Id,Category\n"
    csv.write(columnTitleRow)

    for i in prediction:
        csv.write(str(index) + ',' + str(i) + "\n")
        index += 1

    csv.close()
