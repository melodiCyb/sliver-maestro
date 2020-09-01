from sklearn.preprocessing import MinMaxScaler
from src.utils.im_utils import *
from configparser import ConfigParser
import argparse
import os

config = ConfigParser()
config.read('config.cfg')


def adjust_output_images(initial_prefix, transparent_prefix, svg=False):
    """
    Generates black drawing pixels with a transparent background.
    
    initial_prefix: str. path of raw outputs
    transparent_prefix: str. path of generated images
    """
    for t in range(5, 20):
        imgname = '%s_%d.png' % (initial_prefix, t)
        new_name = '%s_%d.png' % (transparent_prefix, t)
        s_img = cv2.imread(imgname)  # , -1)

        # set to 4 channels
        s_img = fourChannels(s_img)
        # remove white background
        s_img = cut(s_img)
        # set background transparent
        s_img = transBg(s_img)
        # img = s_img

        #gray = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY)
        indices = np.where(s_img <= 160)
        s_img[indices] = 0
        indices = np.where(s_img > 160)
        s_img[indices] = 255

        img = cv2.bitwise_not(s_img)

        cv2.imwrite(new_name, img)
        img = cv2.imread(new_name)
        cv2.imwrite(new_name, img)

    if svg:
        for t in range(5, 20):
            # alternative online-converter
            imgname = '%s_%d.png' % (transparent_prefix, t)
            new_name = '%s_%d.svg' % (transparent_prefix, t)
            s_img = cv2.imread(imgname)

            #img = cv2.bitwise_not(s_img)
            plt.imsave(new_name, s_img, format='svg')


def generate_motion(svg_to_csv_base_path, scaled_base_path, final_motion):
    # Prepare for V-rep
    def scale_coordinates(svg_to_csv_base_path, scaled_base_path):
        """
        Scales X and Y coordinates for robot scene.
        """
        for i in range(17, 20):
            path = '%s_%d.csv' % (svg_to_csv_base_path, i)
            output_path = '%s_%d.csv' % (scaled_base_path, i)
            simple_co = np.genfromtxt(path, delimiter=',', skip_header=1,
                                      usecols=(1, 2, 3), dtype=np.float)
            scaler_x = MinMaxScaler(feature_range=(-0.4, 0.4))
            scaler_x.fit(simple_co[:, 0].reshape(-1, 1))
            simple_co_scaled_x = scaler_x.transform(simple_co[:, 0].reshape(-1, 1))
            scaler_y = MinMaxScaler(feature_range=(-0.2, 0.2))
            scaler_y.fit(simple_co[:, 1].reshape(-1, 1))
            simple_co_scaled_y = scaler_y.transform(simple_co[:, 1].reshape(-1, 1))

            df = pd.read_csv(path, index_col=0)
            df_scaled = df.copy(deep=True)

            df_scaled['X(m)'] = simple_co_scaled_x
            df_scaled['Y(m)'] = simple_co_scaled_y
            df_scaled['Z(m)'] = df_scaled['Z(m)'].apply(lambda x: 0.0 if x < 1 else 0.006)
            df_scaled.to_csv(output_path)

    def join_dframes(scaled_base_path, final_motion):
        """
        Joins frames that consist to ordered pixels in top and bottom part of the image
        base_merge_path: str. path for merged dataframes
        """

        full_frame = pd.DataFrame(columns=['X(m)', 'Y(m)', 'Z(m)'])
        for j in range(17, 20):
            path = '%s_%d.csv' % (scaled_base_path, j)
            df = pd.read_csv(path, index_col=0)
            full_frame = full_frame.append(df)

        seconds = [0.0]
        s = 0.0
        for i in range(len(full_frame)):
            s += 0.05
            seconds.append(s)

        full_frame['Seconds'] = seconds[:-1]
        full_frame = full_frame.set_index('Seconds')
        full_frame.to_csv(final_motion)

    scale_coordinates(svg_to_csv_base_path, scaled_base_path)
    join_dframes(scaled_base_path, final_motion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sliver-maestro')
    parser.add_argument('-rp', '--rootpath')
    args = parser.parse_args()
    root_path = args.rootpath
    if not root_path:
        root_path = os.getcwd()

    initial_prefix = os.path.join(root_path, config['adjust_output_images']['initial_prefix'])
    transparent_prefix = os.path.join(root_path, config['adjust_output_images']['transparent_prefix'])
    adjust_output_images(initial_prefix=initial_prefix, transparent_prefix=transparent_prefix, svg=False)

    svg_base_path = os.path.join(root_path, config['SVG']['svg_base_path'])
    svg_to_csv_base_path = os.path.join(root_path, config['SVG']['svg_to_csv_base_path'])

    for i in range(17, 20):
        svg_file = '%s_%d.svg' % (svg_base_path, i)
        csv_path = '%s_%d.csv' % (svg_to_csv_base_path, i)
        parse_svg(svg_file=svg_file, csv_path=csv_path)

    scaled_base_path = os.path.join(root_path, config['generate_motion']['scaled_base_path'])
    final_motion = os.path.join(root_path, config['generate_motion']['final_motion'])
    generate_motion(svg_to_csv_base_path=svg_to_csv_base_path, scaled_base_path=scaled_base_path,
                    final_motion=final_motion)
