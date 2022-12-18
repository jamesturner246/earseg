import os
import argparse
import PIL.Image
import numpy as np
import cv2


def convert_crop(
        image_path, label_path,
        crop_size=(512, 512),
        crop_offset=(0, 0),
        image_out_dir="png_image",
        label_out_dir="png_label",
        use_remainder=False,
        **kwargs):

    print(f"Image path: {image_path}")
    print(f"Label path: {label_path}")

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    label_name = os.path.splitext(os.path.basename(label_path))[0]
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    with PIL.Image.open(image_path) as image, PIL.Image.open(label_path) as label:
        assert(image.size == label.size)
        image_size = image.size
        crop_size = tuple(crop_size)
        crop_offset = tuple(crop_offset)

        for x in range(crop_offset[0], image_size[0], crop_size[0]):
            x_next = x + crop_size[0]
            if x_next > image_size[0]:
                if use_remainder:
                    x_next = image_size[0]
                else:
                    continue

            for y in range(crop_offset[1], image_size[1], crop_size[1]):
                y_next = y + crop_size[1]
                if y_next > image_size[1]:
                    if use_remainder:
                        y_next = image_size[1]
                    else:
                        continue

                image_out = image.crop((x, y, x_next, y_next))
                image_out_file = f"{image_name}_{x}_{y}_{x_next}_{y_next}.png"
                image_out_path = os.path.join(image_out_dir, image_out_file)
                image_out.save(image_out_path, optimize=False, compress_level=0)

                label_out = label.crop((x, y, x_next, y_next))
                label_out_file = f"{label_name}_{x}_{y}_{x_next}_{y_next}.png"
                label_out_path = os.path.join(label_out_dir, label_out_file)
                label_out.save(label_out_path, optimize=False, compress_level=0)

                print(f"x: [{x}, {x_next}]  w: {x_next - x}")
                print(f"y: [{y}, {y_next}]  h: {y_next - y}")
                print()

                # cv2.imshow("Image Crop", np.array(image_out))
                # cv2.imshow("Label Crop", np.array(label_out))
                # k = cv2.waitKey(0)
                # if k == ord("q"):
                #     cv2.destroyAllWindows()
                #     return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("label_path", type=str)
    parser.add_argument("--crop-size", type=int, default=[512, 512], nargs=2)
    parser.add_argument("--crop-offset", type=int, default=[0, 0], nargs=2)
    parser.add_argument("--image-out-dir", type=str, default="png_image")
    parser.add_argument("--label-out-dir", type=str, default="png_label")
    parser.add_argument("--use-remainder", action="store_true")

    args = parser.parse_args()
    convert_crop(**vars(args))
