# This python script is intended to find the points next to the ring finger and the distance between them

## Usage

You need to activate python virtualenv with `source venv/bin/activate` beforehand.

```
usage: finger_finder.py [-h] --img-path IMG_PATH -url
```
```
optional arguments:
  -h, --help           show this help message and exit
  --img-path IMG_PATH  Path of image
  -url
  -path
```
**Note**: The script works for single image, so the argument must be the full path of the image.

    python3 ./finger_finder.py  <input_image_path>
    
*Example*

### If your path is local
```
python3 ./finger_finder.py -path --img-path ./tests/test.png
```

### If your path is url
```
python3 ./finger_finder.py -url --img-path 'https://stanfordflipside.com/images/45Hand.jpg'
```

**Please Note**: Script works only for those images in which there is one hand and it is straight towards the camera. The script does not support rotations.

## Contact

For questions please contact Ani Vardanyan at `vardanyan.ani28@gmail.com`.


