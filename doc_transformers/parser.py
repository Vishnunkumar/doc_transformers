import pandas as pd
import numpy as np
import pytesseract
import torch
from itertools import groupby
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont

datasets = load_dataset("nielsr/funsd")
labels = datasets['train'].features['ner_tags'].feature.names
id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

def load_models():
  
  processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
  model = LayoutLMv3ForTokenClassification.from_pretrained(
      "nielsr/layoutlmv3-finetuned-funsd"
  )
  
  return processor, model

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

def process_image(image, processor, model):
  
  width, height = image.size
  encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
  offset_mapping = encoding.pop("offset_mapping")

  outputs = model(**encoding)

  predictions = outputs.logits.argmax(-1).squeeze().tolist()
  token_boxes = encoding.bbox.squeeze().tolist()

  is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0
  true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
  true_boxes = [
      unnormalize_box(box, width, height)
      for idx, box in enumerate(token_boxes)
      if not is_subword[idx]
  ]
  
  true_boxes = true_boxes[1:-1]
  true_predictions = true_predictions[1:-1]

  preds = []
  l_words = []
  bboxes = []

  for i,j in enumerate(true_predictions):

    if j != 'O':
      preds.append(true_predictions[i])
      l_words.append(words[0][i])
      bboxes.append(true_boxes[i])

  return bboxes, preds, l_words, image

def iob_to_label(label):
  label = label[2:]
  if not label:
    return 'other'
  return label

def visualize_image(final_bbox, final_preds, l_words, image):

  draw = ImageDraw.Draw(image)
  font = ImageFont.load_default()
  
  label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}
  l2l = {'question':'key', 'answer':'value', 'header':'title'}
  f_labels = {'question':'key', 'answer':'value', 'header':'title', 'other':'others'}

  json_df = []

  for ix, (prediction, box) in enumerate(zip(final_preds, final_bbox)):
    predicted_label = iob_to_label(prediction).lower()
    draw.rectangle(box, outline=label2color[predicted_label])
    draw.text((box[0] + 10, box[1] - 10), text=f_labels[predicted_label], fill=label2color[predicted_label], font=font)

    json_dict = {}
    json_dict['TEXT'] = l_words[ix]
    json_dict['LABEL'] = f_labels[predicted_label]
    
    json_df.append(json_dict)

  return image, json_df

def process_form(json_df):

  labels = [x['LABEL'] for x in json_df]
  texts = [x['TEXT'] for x in json_df]
  cmb_list = []
  for i, j in enumerate(labels):
    cmb_list.append([labels[i], texts[i]])
    
  grouper = lambda l: [[k] + sum((v[1::] for v in vs), []) for k, vs in groupby(l, lambda x: x[0])]
  
  list_final = grouper(cmb_list)
  lst_final = []
  for x in list_final:
    json_dict = {}
    json_dict[x[0]] = (' ').join(x[1:])
    lst_final.append(json_dict)
  
  return lst_final
