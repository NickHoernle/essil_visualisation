import numpy as np
import cv2
import pylab as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import classification_evaluation as hdp_eval
import sys

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg' # Add the path of ffmpeg here!!


# thanks to Aditi for the OpenCV code for processing the global view
import opencv_utils

matplotlib.font_manager._rebuild()


def get_water_representation(min_video_frame):
    boundaries, boundary_ends, center_biomes = opencv_utils.mark_boundaries()

    t = 150
    min_video_frame[(min_video_frame[:, :, 0] > t) & (min_video_frame[:, :, 1] > t) & (min_video_frame[:, :, 2] > t)] = np.zeros(3, dtype=np.uint8)
    dst = np.max([boundaries.astype(np.uint8), min_video_frame], axis=0)

    lower=np.array([0, 64, 60])
    upper=np.array([10, 115, 109])
    dst = opencv_utils.fix_color(dst, lower, upper, np.array([66, 244, 232], dtype=np.uint8), roll=1)

    lower=np.array([84, 0, 5])
    upper=np.array([246, 55, 50])
    dst = opencv_utils.fix_color(dst, lower, upper, np.array([255, 0, 0], dtype=np.uint8), roll=1)

    lower=np.array([0, 0, 0])
    upper=np.array([100, 100, 100])
    dst = opencv_utils.fix_color(dst, lower, upper, np.array([255, 255, 255], dtype=np.uint8), roll=0)

    return dst


def create_black_border(thickness, dimensions, data):
    frame = np.ones(shape=(dimensions[0]+2*thickness, dimensions[1]+2*thickness, 3), dtype=np.uint8)*255
    frame[thickness:-thickness, thickness:-thickness, :] = data
    frame[:5, :, :] = np.array([190, 192, 196])
    frame[:, -6:, :] = np.array([190, 192, 196])
    frame[:, :5, :] = np.array([190, 192, 196])
    frame[-4:, :, :] = np.array([190, 192, 196])
    return frame


def rotate_image(frame, rotation=90):
    data = frame
    height, width = data.shape[:2]
    image_center = (width/2, height/2)
    rotation_mat = cv2.getRotationMatrix2D((width/2, height/2),rotation,1)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    data = cv2.warpAffine(data, rotation_mat, (height, width))
    return data


def transform(frame, dimensions):
    data = get_water_representation(frame)
    data = create_black_border(25, dimensions, data)
    data = rotate_image(data, 90)
    return data


def add_annotations(ax, fontsize=10, text_col='black'):

    ann = ax.annotate('Grasslands', xy=[0, 0], xytext=[10, 50], rotation=40, fontsize=fontsize, color=text_col, fontname='Arial', weight='bold')
    ann = ax.annotate('River Valley', xy=[0, 0], xytext=[10, 150], rotation=90, fontsize=fontsize, color=text_col, fontname='Arial', weight='bold')
    ann = ax.annotate('Desert', xy=[0, 0], xytext=[10, 280], rotation=90, fontsize=fontsize, color=text_col, fontname='Arial', weight='bold')

    ann = ax.annotate('Jungle', xy=[0, 0], xytext=[175, 100], rotation=-55, fontsize=fontsize, color=text_col, fontname='Arial', weight='bold')
    ann = ax.annotate('Reservoir', xy=[0, 0], xytext=[230, 200], rotation=-90, fontsize=fontsize, color=text_col, fontname='Arial', weight='bold')
    ann = ax.annotate('Wetlands', xy=[0, 0], xytext=[230, 300], rotation=-90, fontsize=fontsize, color=text_col, fontname='Arial', weight='bold')

    ann = ax.annotate('Waterfall', xy=[0, 0], xytext=[110, 20], rotation=0, fontsize=fontsize, color=text_col, fontname='Arial', weight='bold')
    # return ax


def get_file_images(in_file_name, out_folder_name):

    cap = cv2.VideoCapture(in_file_name)

    count = 0
    ret = True
    while ret:
        ret, frame = cap.read()

        if not ret:
            return

        y_min, y_max, x_min, x_max = 0, 384, 864, 1080
        dimensions = x_max-x_min, y_max-y_min

        data = transform(frame[864:1080, 0:384], dimensions)

        fig = plt.figure()
        fig.set_size_inches([4, 5])

        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        add_annotations(ax)

        im = ax.imshow(data)
        plt.savefig(f'{out_folder_name}/img{count}.png', bbox_inches='tight', pad_inches=None)

        if cv2.waitKey(10) == 27:
            break
        count += 1

        plt.close()
        plt.cla()
        plt.clf()
        plt.close(fig)


def process_global_vid(df, cap, out_file):

    count = 0
    dpi = 1000

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ret, frame = cap.read()

    y_min, y_max, x_min, x_max = 0, 384, 864, 1080
    dimensions = x_max-x_min,y_max-y_min

    data = transform(frame[864:1080,0:384], dimensions)
    # data = np.ones_like(data)*255
    im = ax.imshow(data)

    fontSize = 13
    textCol = 'black'

    add_annotations(ax, fontSize, textCol)

    fig.set_size_inches([4,5.5])

    pl.tight_layout()

    def update_img(n):
        ret,frame = cap.read()

        if ret==True:
            data = transform(frame[864:1080,0:384], dimensions)
            # data = np.ones_like(data)*255
    #         data = blur_image('wetlands', data)
            im.set_data(data)

        return im

    length = int(df._video_seconds.max())
    ani = animation.FuncAnimation(fig, update_img, length, interval=500)
    writer = animation.writers['ffmpeg'](fps=1)
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)

    ani.save(out_file)

    cap.release()
