from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/tiny-yolo-voc-1c.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("./sample_img/sample_signal.jpeg")
result = tfnet.return_predict(imgcv)
print(result)

## The command to run:
## python flow --train --model cfg/tiny-yolo-voc-1c.cfg --epoch 5 --lr 1e-2 --batch 128 --keep 500 --gpu 1.0 --annotation test/training/Annotations --dataset test/training/Images
