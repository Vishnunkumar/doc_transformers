# Doc Transformers
Document processing using transformers

## Pre-requisites

Please install the following seperately
```
sudo apt install tesseract-ocr
pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
```

## Implementation

```
# loads the pretrained dataset also 
from doc_transformers import form_parser

# loads the image
image = form_parser.load_image(input_path_image)

# gets the bounding boxes, predictions and image processed
bbox, preds, image = form_parser.process_image(image)

# returns image as the output
im = form_parser.visualize_image(bbox, preds, image)
```

## Results

**Input**

![input image](ik.jpg) 

**Output**

![output image](output.png)

- Please note that this is still in development phase and will be improved in the near future
