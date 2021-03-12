
import time
import warnings
from configparser import ConfigParser
import logging

import tensorflow as tf
from tensorflow.python.ops import ctc_ops

import sys
import argparse
import scipy.cluster.vq as vq
import numpy as np
import glob, os


from trajnorm import normalize_trajectory
from trajfeat import calculate_feature_vector_sequence
# Custom modules
# from text import ndarray_to_text, sparse_tuple_to_texts

# in future different than utils class
from utils import create_optimizer,rearrange
from datasets import pad_sequences,sparse_tuple_from,handwriting_to_input_vector
from set_dirs import get_conf_dir, get_model_dir
import gpu as gpu_tool
# Import the setup scripts for different types of model
from rnn import BiRNN as BiRNN_model
from rnn import SimpleLSTM as SimpleLSTM_model
from IPython.display import HTML, Image
from google.colab.output import eval_js
from base64 import b64decode



letters_ar= {}
with open('Arabic_Mappping.csv',encoding="utf-8") as f:
    for line in f:
       (key, val) = line.split(',')
       letters_ar[int(key)] =val.strip()     
#        print(key)
#        letters_ar[val.strip()] =int(key) 

letters_ar[83]=' '


#LM setting
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
lm_alpha=0.75
lm_beta= 1.85
lm_binary_path='lm/lm.binary'
lm_trie_path='lm/trie'
beam_width=32
cutoff_prob=1.0
cutoff_top_n= 300
scorer=None

from text import Alphabet
alphabet = Alphabet('alphabet.txt')

scorer = Scorer(lm_alpha, lm_beta,
                        lm_binary_path, lm_trie_path,
                        alphabet)

def decodex(txt,mapping):
    out=''
    for ch in txt:
        out=out+mapping[ch]
    return out
mapping = {}
   
with open('arabic_mapping.txt','r', encoding='utf-8') as inf:
    for line in inf:
        key,val=line.split('\t')
        mapping[key]=val.strip()
mapping[' ']=' '


config_file='neural_network.ini'
model_name='model.ckpt-14'
model_path = os.path.join('models', model_name) 
parser = ConfigParser(os.environ)
conf_path = config_file

parser.read('neural_network.ini')
# set which set of configs to import
config_header = 'nn'        
network_type = parser.get(config_header, 'network_type')
# Number of mfcc features, 20
n_input = parser.getint(config_header, 'n_input')
# Number of contextual samples to include
n_context = parser.getint(config_header, 'n_context')       
model_dir = parser.get(config_header, 'model_dir')
# setup type of decoder
beam_search_decoder = parser.get(config_header, 'beam_search_decoder')        
# set up GPU if available
tf_device = str(parser.get(config_header, 'tf_device'))
# set up the max amount of simultaneous users
# this restricts GPU usage to the inverse of self.simultaneous_users_count
simultaneous_users_count = parser.getint(config_header, 'simultaneous_users_count')



input_tensor = tf.placeholder(tf.float32, [None, None, n_input + (2 * n_input * n_context)], name='input')    
seq_length = tf.placeholder(tf.int32, [None], name='seq_length')
logits, summary_op = BiRNN_model(conf_path,input_tensor,tf.to_int64(seq_length),n_input,n_context)  
decoded, log_prob = ctc_ops.ctc_greedy_decoder(logits, seq_length, merge_repeated=True)

saver = tf.train.Saver()
# create the session
sess = tf.Session()
saver.restore(sess, model_path)
print('Model restored') 
canvas_html = """
<canvas id="mycanvas" width=%d height=%d style="border:1px solid #000000;"></canvas>
 <br />
<button>Finish</button>

<script>
var canvas = document.getElementById('mycanvas')
var ctx = canvas.getContext('2d')
ctx.lineWidth = %d
ctx.canvas.style.touchAction = "none";
var button = document.querySelector('button')
var mouse = {x: 0, y: 0}
var points=[]

canvas.addEventListener('pointermove', function(e) {
  mouse.x = e.pageX - this.offsetLeft
  mouse.y = e.pageY - this.offsetTop
})
canvas.onpointerdown = ()=>{
  ctx.beginPath()
  ctx.moveTo(mouse.x, mouse.y)
  
  canvas.addEventListener('pointermove', onPaint)
}
canvas.onpointerup = ()=>{
  canvas.removeEventListener('pointermove', onPaint)
  points.pop()
  points.push([mouse.x,mouse.y,1])
}
var onPaint = ()=>{
  ctx.lineTo(mouse.x, mouse.y)
  ctx.stroke()
  points.push([mouse.x,mouse.y,0])
}
var data = new Promise(resolve=>{
  button.onclick = ()=>{
    resolve(canvas.toDataURL('image/png'))
  }
})
</script>
"""
def draw(filename='drawing.png', w=900, h=200, line_width=1):
  display(HTML(canvas_html % (w, h, line_width)))
  data = eval_js("data")
  points=eval_js("points")
  # strokes = Utils.Rearrange(strokes, 20);
  points=np.array(points)

  # points=rearrange(points)


  # print("Points before pre",points.shape)
  NORM_ARGS = ["origin","filp_h","smooth", "slope", "resample", "slant", "height"]
  FEAT_ARGS = ["x_cor","y_cor","penup","dir", "curv", "vic_aspect", "vic_curl", "vic_line", "vic_slope", "bitmap"]
  # print("Normalizing trajectory...")
  traj = normalize_trajectory(points, NORM_ARGS)
  # print(traj)
  # print("Calculating feature vector sequence...")
  feat_seq_mat = calculate_feature_vector_sequence(traj, FEAT_ARGS)
  feat_seq_mat=feat_seq_mat.astype('float32')
  feat_seq_mat.shape

  data = []

  train_input=handwriting_to_input_vector(feat_seq_mat,20,9)
  train_input = train_input.astype('float32')

  data.append(train_input)
  # data_len

  data = np.asarray(data)
  # data_len = np.asarray(train_input)


  # Pad input to max_time_step of this batch
  source, source_lengths = pad_sequences(data)
  my_logits=sess.run(logits, feed_dict={
                  input_tensor: source,                    
                  seq_length: source_lengths}
              )
  my_logits = np.squeeze(my_logits)
  maxT, _ = my_logits.shape # dim0=t, dim1=c
	
            # apply softmax
  res = np.zeros(my_logits.shape)
  for t in range(maxT):
      y = my_logits[t, :]
      e = np.exp(y)
      s = np.sum(e)
      res[t, :] = e / s
            
  decoded = ctc_beam_search_decoder(res, alphabet, beam_width,
                                  scorer=scorer, cutoff_prob=cutoff_prob,
                                  cutoff_top_n=cutoff_top_n)

  print("Result : "+decodex(decoded[0][1],mapping))
  # d = sess.run(decoded[0], feed_dict={
  #                 input_tensor: source,                    
  #                 seq_length: source_lengths}
  #             )
  # dense_decoded = tf.sparse_tensor_to_dense( d, default_value=-1).eval(session=sess)        
  # print(dense_decoded[0]) 
  # print(''.join([letters_ar[label] for label in dense_decoded[0].tolist() ]))  
  # print(points)
  # return np.array(points)