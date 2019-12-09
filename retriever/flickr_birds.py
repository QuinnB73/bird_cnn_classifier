import flickrapi
import lxml.etree as etree
import os
from argparse import ArgumentParser

MAX_PHOTOS = 1200
BIRDS = [
    'black-capped chickadee'
    'downy woodpecker',
    'dark-eyed junco',
    'blue jay',
    'white-breasted nuthatch',
    'mourning dove',
    'american goldfinch',
    'northern cardinal',
    'red-breasted nuthatch',
    'european starling',
    'house sparrow',
    'house finch',
    'american robin',
    'common grackle',
    'red-winged blackbird',
    'american crow',
    'american tree sparrow',
    'red-bellied woodpecker',
    'common redpoll',
    'song sparrow',
    'pine siskin',
    'cooper\'s hawk',
    'brown-headed cowbird',
    'evening grosbeak'
]

argparser = ArgumentParser()
argparser.add_argument('-o', '--output-dir', help='The output directory for images of birds')

def get_url(etree_fphoto):
    """ This function builds a Flickr URL for an image based on its metadata."""

    photo_id = etree_fphoto.get('id')
    server = etree_fphoto.get('server')
    secret = etree_fphoto.get('secret')
    farm = etree_fphoto.get('farm')
    return f'https://farm{farm}.staticflickr.com/{server}/{photo_id}_{secret}.jpg'

def save_url_file_for_bird(flickr, keyword, output_dir):
    """ This function writes all built URLs to a file for a given keyword."""

    filename = f'{output_dir}/{keyword}.urls.txt'
    filename = filename.replace(' ', '-')
    print(f'Getting URLs for {keyword} and saving to file {filename}')

    count = 0
    with open(filename, 'w+') as url_file:
        for photo in flickr.walk(
                tag_mode='all',
                tags=keyword,
                content_type=1):
            url = get_url(photo)
            url_file.write(f'{url}\n')
            count += 1

            # There is no way in flickr.walk to limit number of pages, so do it
            # manually, since it's a generator there should be no issues
            if count >= MAX_PHOTOS:
                break

def main():
    API_KEY = os.getenv('FLICKR_API_KEY').encode('utf-8')
    API_SECRET = os.getenv('FLICKR_SECRET').encode('utf-8')
    args = argparser.parse_args()
    output_dir = args.output_dir

    flickr = flickrapi.FlickrAPI(API_KEY, API_SECRET, format='etree')

    for bird in BIRDS:
        save_url_file_for_bird(flickr, bird, output_dir)

    print('Done')

if __name__ == "__main__":
    main()