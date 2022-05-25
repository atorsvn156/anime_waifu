from gtts import gTTS
import subprocess
import re
import sys, math, numpy, wave, struct
import scipy
from scipy.fftpack import rfft
import numpy as np
import tflite_runtime.interpreter as tflite
import os
import cv2
import discord
from discord import File
import requests
import json


IS_IDLE = True
idle_index = 0
client = discord.Client()
interpreter =  tflite.Interpreter(model_path="./model/waifu.tflite", num_threads=1)
interpreter.allocate_tensors()

with open('config/waifu_config.json') as json_file:
    CONFIG = json.load(json_file)
  
async def add_vid_subs(vid_in, query, vid_out):
    with open("subs.srt", "w") as myfile:
        myfile.write("0\n00:00:00,000 --> 01:00:00,000\n"+query)
    subprocess.call(['ffmpeg', '-i', vid_in, '-vf', 'subtitles=subs.srt', '-r', '12.5','-vcodec', 'libx264', vid_out])

def cleanup():
    os.system("rm output.mp4")
    os.system("rm response.wav")
    os.system("rm waifu.mp4")

def waifu_query(query, user_id, user_name, server_name, channel_name):
    url = "https://waifu.p.rapidapi.com/path"

    querystring = {"user_id":user_id,"message":query,"from_name":user_name,"to_name":CONFIG['waifu-name'],"situation": CONFIG['situation'],"translate_from":"auto","translate_to":"auto"}

    my_obj = {
        "key1": "value",
        "key2": "value"
    }
    
    payload = json.dumps(my_obj)

    headers = {
    'content-type': "application/json",
    'x-rapidapi-host': "waifu.p.rapidapi.com",
    'X-RapidAPI-Key': CONFIG['rapid-api-key']
    }
    
    response = requests.request("POST", url, data=payload, headers=headers, params=querystring)
    
    reply = response.text

    if reply == """
    <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<title>500 Internal Server Error</title>
<h1>Internal Server Error</h1>
<p>The server encountered an internal error and was unable to complete your request. Either the server is overloaded or there is an error in the application.</p>
    """:
        reply = "Hold on just a second baby. Gotta do something."

    return str(reply)


def model_predict(d):
    input_details = interpreter.get_input_details()
    input_data = np.array(d, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    return(np.argmax(predictions))

capture = cv2.VideoCapture('assets/waifu_work2.mp4')  # reading the video
success, image = capture.read()
images = []  # creating empty list to store frames
n_frame = 1


face_model = {
    "0" : []
}

while success:
    #image = cv2.resize(image, (512, 512))  # resizing the frame to 256x256
    images.append(image)  # appending it to images list defined above
    face_model[str(n_frame)] = image
    n_frame += 1
    success, image = capture.read()  # reading a frame

HEIGHT = 100 # height of a bar
WIDTH = 10 # width of a bar
FPS = 25#25

async def do_avatar(response_text):
    idle_index = 0
    file_name = "response.wav"

    # process wave data
    f = wave.open(file_name, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data  = f.readframes(nframes)  
    f.close()  
    wave_data = np.frombuffer(str_data, dtype = np.short)  
    wave_data.shape = -1,2  
    wave_data = wave_data.T  

    N = 30 # num of bars
    num = nframes
   
    the_switch = False

    resolution = (512, 512)
    codec = cv2.VideoWriter_fourcc(*'mp4v') 
    filename = "movie.mp4"

    out = cv2.VideoWriter(filename, codec, 12.5, resolution)
    
    last_prediction = 0

    for count in range(int(nframes/float(framerate/12.5))):#(framerate/FPS))):
        h_old = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



        num -= framerate/FPS
        num = int(num)
        try:
            h = abs(rfft(wave_data[0][nframes - num:nframes - num + N]))
        except:
            h = h_old
        
        h = np.abs([min(HEIGHT,int(i **(1 / 2.5) * HEIGHT / 100)) for i in h])
        n = h.size // 8    # 32 frequency bands
        bands = [sum(h[i:(i + n)]) for i in range(0, h.size, n)]
        test = np.array([bands])
        #print(test)
        cutoff = 25

        prediction = model_predict([[0,0,0,0,0,0,0,0,0,0]])
        if np.any(h >= cutoff):
            prediction = model_predict(test)

        fram = face_model[str(prediction)]
        if(fram == []):
            fram = face_model[str(1)]
        out.write(fram)
    out.release()
    subprocess.run(["ffmpeg", "-i", "movie.mp4", "-i", "response.wav", "-map", "0:v:0", "-map", "1:a:0", "-r", "12.5", "-vcodec", "libx264", "output.mp4"])
    await add_vid_subs("output.mp4", response_text,"waifu.mp4")
    return 23

#board = pyfirmata.Arduino('COM5')
#print("Communication Successfully started")
    
@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(msg):
    if msg.author == client.user:
        return
    if msg.content.startswith(CONFIG['waifu-command']):
        query = re.sub(CONFIG['waifu-command'] + " ", '', msg.content)
        await play_response(query, msg)
      
async def do_tts(query): 
    cleanup()
    response_text = query #waifu_query(query)
    
    tts = gTTS(response_text,lang='en')
    tts.save('response.mp3')
    subprocess.run(["ffmpeg", "-i", "response.mp3", "response.wav"])
    subprocess.run(["sox", "response.wav", "temp.wav", "tempo", "1.2"])
    subprocess.run(["sox", "temp.wav", "response.wav", "bend", ".10,400,.25"])

    await do_avatar(response_text)
    #await message.channel.send("```"+response_text+"```")

async def play_response(query, message):
    global IS_IDLE
    IS_IDLE = False

    response = waifu_query(query, str(message.author.id), str(message.author.name), str(message.guild.name), str(message.channel.name))
    await do_tts(response)


    await message.channel.send(file=discord.File('waifu.mp4'))

client.run(CONFIG['discord-token'])

