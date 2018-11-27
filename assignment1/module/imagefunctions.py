from PIL import Image

def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


def load_image(path):
    image = Image.open(path)
    global image_dimensions
    image_dimensions = image.width, image.height
    return image


def flatten_image(image):
    """
    turn image into list of [r,g,b]
    :param image:
    :return:
    """
    x_w, y_w = image.width, image.height
    data = list()
    for x in range(0, x_w):
        for y in range(0, y_w):
            vector_data = list(get_rgb_color(image, x, y))
            data.append(vector_data)
    return data


def unflatten_image(image, clustering, k):
    """
    recolor image based on cluster id
    :param image: Pillow image, that was worked on
    :param clustering:
    :param k: max cluster_id -1
    :return: the colored version of the image
    """
    x_w, y_w = image.width, image.height
    colors = get_spaced_colors(k)
    list_pos = 0
    for x in range(0, x_w):
        for y in range(0, y_w):
            cluster_id = clustering[list_pos]
            set_rgb_color(image, x, y, colors[cluster_id])
            list_pos += 1
    return image


def save_image(image, path="output.bmp", outformat="BMP"):
    image.save(path, outformat)


def get_rgb_color(image, x, y):
    return image.getpixel((x, y))


def set_rgb_color(image, x, y, (r, g, b)):
    image.putpixel((x, y), (r, g, b))