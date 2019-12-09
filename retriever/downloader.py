import argparse
import requests
import sys
import os

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-u',
    '--urls_dir',
    help='path to image file containing URLs'
)
argparser.add_argument(
    '-i',
    '--images-dir',
    help='path to the output directory of the images'
)

def download_images(urls_dir, images_dir):
    """ This function downloads images from the URLs into the provided
    images dir."""

    bird_urls = read_urls_file(urls_dir)

    for bird, urls in bird_urls.items():
        counter = 0
        try:
            os.mkdir(f'{images_dir}/{bird}')
        except OSError:
            print(f'Unable to create images directory for bird {bird}. Exiting.')
            sys.exit(2)

        num_images = len(urls)

        print(f'Will attempt to download {num_images} for {bird}')
        success = 0
        for url in urls:
            try:
                if counter % 10 == 0:
                    print(f'{bird}: Attempted to download {counter} images so far. Successfully downloaded {success}.')

                downloaded_image = requests.get(url, timeout=60)
                if downloaded_image.status_code >= 300:
                    continue

                fileName = f'{images_dir}/{bird}/{counter}.png'
                with open(fileName, 'wb') as f:
                    f.write(downloaded_image.content)
                success += 1
            except:
                pass
            finally:
                counter += 1
        error_amount = num_images - success
        print(f'Successfully downloaded {success} images for {bird}. Unable to download {error_amount} images.')

def read_urls_file(urls_dir):
    """ This function reads the URL files in the provided directory."""

    bird_urls = {}
    for _, __, fileList in os.walk(urls_dir):
        for fileName in fileList:
            fileName = f'{urls_dir}/{fileName}'
            print(f'Opening file: {fileName}')
            with open(fileName, 'r') as f:
                bird = fileName.replace(f'{urls_dir}/', '')
                bird = bird.replace('.urls.txt', '')
                bird_urls[bird] = [k for k in f]
        
    return bird_urls

def main():
    args = argparser.parse_args()
    download_images(args.urls_dir, args.images_dir)

if __name__ == "__main__":
    main()
