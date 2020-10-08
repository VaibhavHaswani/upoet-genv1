import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL #suppress unusual warning
logging.getLogger('tensorflow').setLevel(logging.FATAL)


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


data=open('./data/irish_poem.txt','r').read()
corpus=data.lower().split("\n")
tokenizer=Tokenizer()
tokenizer.fit_on_texts(corpus)

model=tf.keras.models.load_model('models/poemgenmodel.h5')

size=int(input("Enter size of poem to generate(words):"))
start=input("Enter Starting Words for your poem:")
decoder=dict([(v,k) for (k,v) in tokenizer.word_index.items()])
next_words=size-len(start.split())
count=0
print("\n...Generating Poem...\n")
for _ in range(next_words):
    tokens=tokenizer.texts_to_sequences([start])[0]
    pad=pad_sequences([tokens],maxlen=15)
    word_token=model.predict_classes(pad,verbose=0)[0]
    #print(word_token)
    word=decoder[word_token]
    start+=" "+word
    if count>10:
        count=0
        start+=",\n"
    count+=1
print(start+"...")
