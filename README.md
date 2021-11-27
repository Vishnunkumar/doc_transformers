# Doc Transformers
Document processing using transformers. This is still in developmental phase, currently supports only extraction of form data i.e (key - value pairs)

```
pip install -q doc-transformers
```

## Pre-requisites

Please install the following seperately
```
sudo apt install tesseract-ocr
pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
```

## Implementation

```python
# loads the pretrained dataset also 
from doc_transformers import form_parser

# loads the image
image = form_parser.load_image(input_path_image)

# gets the bounding boxes, predictions, extracted words and image processed
bbox, preds, words, image = form_parser.process_image(image)

# returns image and extracted key-value pairs along with title as the output
im, df = form_parser.visualize_image(bbox, preds, words, image)

# process and returns k-v pairs by concatenating relevant strings.
df_main = form_parser.process_form(df)
```

## Results

**Input & Output**

<p float="left">
<img src="/bill7.png" width="350" height="600">
<img src="/output.png" width="350" height="600">
</p>

**Table**

- After saving to csv the result looks like the following

| LABEL | TEXT                               |
| ----- | ---------------------------------- |
| title | CREDIT CARD VOUCHER ANY RESTAURANT |
| title | ANYWHERE                           |
| key   | DATE:                              |
| value | 02/02/2014                         |
| key   | TIME:                              |
| value | 11:11                              |
| key   | CARD                               |
| key   | TYPE:                              |
| value | MC                                 |
| key   | ACCT:                              |
| value | XXXX XXXX XXXX                     |
| value | 1111                               |
| key   | TRANS                              |
| key   | KEY:                               |
| value | HYU8789798234                      |
| key   | AUTH                               |
| key   | CODE:                              |
| value | 12345                              |
| key   | EXP                                |
| key   | DATE:                              |
| value | XX/XX                              |
| key   | CHECK:                             |
| value | 1111                               |
| key   | TABLE:                             |
| value | 11/11                              |
| key   | SERVER:                            |
| value | 34                                 |
| value | MONIKA                             |
| key   | Subtotal:                          |
| value | $1969                              |
| value | .69                                |
| key   | Gratuity: Total:                   |

## Code credits

[@HuggingFace](https://huggingface.co/)

- Please note that this is still in development phase and will be improved in the near future
