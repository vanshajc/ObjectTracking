from pydarknet import Detector, Image
import cv2
import time
import tracker
import os

net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0,
               bytes("cfg/coco.data", encoding="utf-8"))


def process_image(img):
    img2 = Image(img)
    results = net.detect(img2)
    return results


def draw_bounds(img, bounds):
    x, y, w, h = bounds
    cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)


def videos():
    cap = cv2.VideoCapture('images/video1.avi')

    objs = tracker.Tracker()

    while True:
        r, img = cap.read()
        if r:
            print("=" * 40)
            start_time = time.time()
            results = process_image(img)
            end_time = time.time()

            for centroid in objs.predict():
                cv2.circle(img, centroid, 10, 0, thickness=2)

            for cat, conf, bounds in results:
                if conf < 0.90:
                    continue
                draw_bounds(img, bounds)
                objs.add(bounds)
            track_time = time.time()
            print("Total Time:", end_time - start_time, track_time - end_time)

            cv2.imshow("preview", img)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    paths = os.listdir('data/single')
    paths.sort()
    imgs = [cv2.imread('data/single/' + f) for f in paths]

    objs = tracker.Tracker()
    num_frames = 0
    for img in imgs:
        print("="*40)
        start_time = time.time()
        results = process_image(img)
        end_time = time.time()

        for centroid in objs.predict():
            cv2.circle(img, centroid, 10, 0, thickness=2)

        print("Total Time:", end_time - start_time)
        for cat, conf, bounds in results:
            if conf < 0.90:
                continue
            draw_bounds(img, bounds)
            objs.add(bounds)

        cv2.imshow("preview", img)

        track_time = time.time()
        print("Total Time:", end_time - start_time, track_time - end_time)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        num_frames += 1
        # if num_frames > 5:
        #     break
    # videos()
