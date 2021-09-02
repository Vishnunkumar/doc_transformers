from datasets import load_dataset 
from PIL import Image, ImageDraw, ImageFont
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2TokenizerFast
import pytesseract
import torch
from transformers import LayoutLMv2ForTokenClassification

datasets = load_dataset("nielsr/funsd")
labels = datasets['train'].features['ner_tags'].feature.names
id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

def load_image(path):
  
  image = Image.open(path)
  image = image.convert("RGB")
  
  return image

def unnormalize_box(bbox, width, height):
 
  return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def process_image(image):
  
  feature_extractor = LayoutLMv2FeatureExtractor.from_pretrained("microsoft/layoutlmv2-base-uncased")
  tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")

  encoding_feature_extractor = feature_extractor(image, return_tensors="pt")
  words, boxes = encoding_feature_extractor.words, encoding_feature_extractor.boxes

  encoding = tokenizer(words, boxes=boxes, return_offsets_mapping=True, return_tensors="pt")
  offset_mapping = encoding.pop('offset_mapping')
  encoding["image"] = encoding_feature_extractor.pixel_values

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  for k,v in encoding.items():
    encoding[k] = v.to(device)

  # load the fine-tuned model from the hub
  model = LayoutLMv2ForTokenClassification.from_pretrained("nielsr/layoutlmv2-finetuned-funsd")
  model.to(device)

  # forward pass
  outputs = model(**encoding)

  predictions = outputs.logits.argmax(-1).squeeze().tolist()
  token_boxes = encoding.bbox.squeeze().tolist()

  width, height = image.size
  final_bbox = []
  final_preds = []

  for i, j in enumerate(predictions):
    if j not in [0, 1, 2]:
      final_bbox.append(unnormalize_box(token_boxes[i], width, height))
      final_preds.append(id2label[j])

  return final_bbox, final_preds, image

def iob_to_label(label):
  label = label[2:]
  if not label:
    return 'other'
  return label

def visualize_image(final_bbox, final_preds, image):

  draw = ImageDraw.Draw(image)
  font = ImageFont.load_default()
  
  label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}

  for prediction, box in zip(final_preds, final_bbox):
      predicted_label = iob_to_label(prediction).lower()
      draw.rectangle(box, outline=label2color[predicted_label])
      draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

  return image
