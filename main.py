from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
import numpy as np
import cv2
import datetime


imageParts = []

# Image Processing function, i is an extra argument i used for the process.
def processBlock(arr, i):
    block = np.frombuffer(arr.get_obj())
    block = block.reshape((512, 512, 3))
    block = block.astype('float32')
    block[:] = cv2.medianBlur(block, 5)
    block = block.reshape((-1))
    arr[:] = block

    print("proses pada core ke " + str(i))


if __name__ == '__main__':
    path = "noise/sample_noise2.jpg"
    fullImage = cv2.imread(path, 1)
    fullImage = cv2.resize(fullImage, (1024, 1024))

    cv2.imshow("Original", fullImage)

    imageParts = []

    # Divide it into 4 parts
    imageParts.append(Array('d', fullImage[0:512, 512:, :].flatten()))
    imageParts.append(Array('d', fullImage[0:512, 0:512, :].flatten()))
    imageParts.append(Array('d', fullImage[512:, 0:512, :].flatten()))
    imageParts.append(Array('d', fullImage[512:, 512:, :].flatten()))

    core = 4

    now = datetime.datetime.now()

    processes = []
    for i in range(core):  # Process each part simulatinously
        p = Process(target=processBlock, args=(imageParts[i], i))
        p.start()
        processes.append(p)

    
    # cv2.imshow("Original1", fullImage)

    for i in range(core):  # Wait for all
        processes[i].join()
    end = datetime.datetime.now()
    delta = end - now
    print("Time computation = " + str(delta) + " second")
    # Reconstruct Image
    fullImage[0:512, 512:, :] = np.frombuffer(
        imageParts[0].get_obj()).reshape((512, 512, 3))
    fullImage[0:512, 0:512, :] = np.frombuffer(
        imageParts[1].get_obj()).reshape((512, 512, 3))
    fullImage[512:, 0:512, :] = np.frombuffer(
        imageParts[2].get_obj()).reshape((512, 512, 3))
    fullImage[512:, 512:, :] = np.frombuffer(
        imageParts[3].get_obj()).reshape((512, 512, 3))

    ouputname = path.split("/")[1].split(".")
    result_path = "output/denoising_{}.{}".format(ouputname[0], ouputname[1])

    cv2.imshow("Filtered", fullImage)

    cv2.waitKey(0)

    cv2.imwrite(result_path, fullImage)
