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
from doc_transformers import form_parser
image = form_parser.load_image(input_path_image)
bbox, preds, image = form_parser.process_image(image)
im = form_parser.visualize_image(bbox, preds, image)
```

## Results
