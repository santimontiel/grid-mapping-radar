import cv2 as cv
import numpy as np

def plot_bev_cloud_with_gt(img_path: str, anno_path: str) -> None:
    """ Plots a frame of RadarScenes_BEV with its correspondent annotations
    into a OpenCV window.

    Args:
        - img_path:
        - anno_path:
    Returns:
        - None

    Source:
        https://stackoverflow.com/questions/64096953/how-to-convert-yolo-format-bounding-box-coordinates-into-opencv-format
    """

    img = cv.imread(img_path, cv.IMREAD_COLOR)
    img = np.array(img)
    dh, dw, _ = img.shape

    fl = open(anno_path, 'r')
    data = fl.readlines()
    fl.close()

    print(data)

    for dt in data:

        # Split string to float
        cat, x, y, w, h = map(float, dt.split(' '))

        # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
        # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        cv.rectangle(img, (l, t), (r, b), (255, 255, 0), 1)
        cv.putText(
            img,                            # image
            str(int(cat)),                  # text
            (r, t - 2),                     # org: coordinates of bottom-left corner
            cv.FONT_HERSHEY_SIMPLEX,        # font
            0.5,                            # fontscale
            (255, 255, 0),                  # color
            1,                              # thickness
            cv.LINE_AA                      # ??
        )



    cv.imshow("BEV Point Cloud with GT", img)
    cv.waitKey(0)
    cv.destroyAllWindows()