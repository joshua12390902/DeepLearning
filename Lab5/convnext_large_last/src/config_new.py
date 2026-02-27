VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

VOC_IMG_MEAN = [0.453112, 0.431648, 0.398453]
VOC_IMG_STD  = [0.240035, 0.235254, 0.238915]
#VOC_IMG_MEAN = (0.45286129, 0.43170348, 0.39989259)  # RGB
#VOC_IMG_STD = (0.2770844, 0.27359877, 0.2856848)  # RGB


COLORS = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

# network expects a square input of this dimension
# YOLO v3 standard is 416x416 (divisible by 32)
YOLO_IMG_DIM = 416
ANCHORS = [
    [(0.0577, 0.0910), (0.1179, 0.2096), (0.3103, 0.2297)],
    [(0.1708, 0.4430), (0.4202, 0.4281), (0.2858, 0.6884)],
    [(0.7997, 0.4600), (0.5284, 0.7764), (0.8744, 0.8638)],
]
GRID_SIZES = [13, 26, 52]