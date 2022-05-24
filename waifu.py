from gtts import gTTS
import subprocess
import os
import re
import sys, math, numpy, wave, struct
import scipy
from scipy.fftpack import rfft
import numpy as np
import tflite_runtime
import cv2
import discord
from discord.ext.commands import MissingPermissions
from discord.ui import Button, View
from discord import File
import json
import requests
import asyncio
import random
from tempfile import NamedTemporaryFile as NTF

IDLE = True
client = discord.Client()
interpreter = tf.lite.Interpreter(model_path="./model/waifu.tflite") #, num_threads=8)
interpreter.allocate_tensors()

with open('config/waifu_config.json') as json_file:
    CONFIG = json.load(json_file)

	
async def add_vid_subs(vid_in, query, vid_out):
	with NTF(mode="w+") as f:
		f.write(f"0\n00:00:00,000 --> 01:00:00,000\n{query}")
		f.flush()
		os.fsync(f.fileno())
		subprocess.run(['ffmpeg', '-y', '-i', vid_in, '-vf', f'subtitles={f.name}', '-r', '12.5','-vcodec', 'libx264', vid_out])

def waifu_query(query, user_id, user_name):
	x = datetime.datetime.now()
	now = x.strftime(" %c. ")

	url = "https://waifu.p.rapidapi.com/path"

	querystring = {
			"user_id" : user_id,
			"message" : query,
			"from_name" : user_name,
			"to_name": CONFIG["waifu-name"],
			"situation": CONFIG["situation"],
			"translate_from":"auto","translate_to":"auto"
	}

	print(querystring["situation"])
	my_obj = {
		"key1": "value",
		"key2": "value"
	}
	
	payload = json.dumps(my_obj)

	headers = {
		'content-type': "application/json",
		'x-rapidapi-host': "waifu.p.rapidapi.com",
		'x-rapidapi-key': CONFIG["rapid-api-key"]

	try:
		response = requests.request("POST", url, data=payload, headers=headers, params=querystring)
		reply = response.text
	except:
		reply = "Hold on just a second baby. Gotta do something real quick."

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
	#image = cv2.resize(image, (512, 512))	# resizing the frame to 256x256
	images.append(image)  # appending it to images list defined above
	face_model[str(n_frame)] = image
	n_frame += 1
	success, image = capture.read()  # reading a frame

HEIGHT = 100 # height of a bar
WIDTH = 10 # width of a bar
FPS = 25#25

async def do_avatar(response_text, response_wav, final_outfile):
	idle_index = 0

	# process wave data
	f = wave.open(response_wav.name, 'rb')
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

	movie = NTF(mode="w+b",suffix=".mp4")

	out = cv2.VideoWriter(movie.name, codec, 12.5, resolution)
	
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

	output = NTF(mode="w+b",suffix=".mp4")
	subprocess.run(["ffmpeg", "-y", "-i", movie.name, "-i", response_wav.name, "-map", "0:v:0", "-map", "1:a:0", "-r", "12.5", "-vcodec", "libx264", output.name])

	await add_vid_subs(output.name, response_text, final_outfile.name)

	output.close()
	movie.close()

	return 23
	
@client.event
async def on_ready():
	print('We have logged in as {0.user}'.format(client))

@client.event
async def on_command_error(ctx, error):
    if isinstance(error, MissingPermissions):
        await ctx.send(":redTick: Make sure bot has permission to send files.")

@client.event
async def on_message(message):
	if message.author == client.user:
		return
	if message.content.startswith(CONFIG["waifu-command"]):
		query = re.sub(CONFIG["waifu-command"] + " ", '', message.content)
		await play_response(query, message)
	  
   
async def do_tts(query, final_outfile): 
	response_text = waifu_query(query)

	response_mp3 = NTF(mode="w+b",suffix=".mp3")
	response_wav = NTF(mode="w+b",suffix=".wav")
	temp_wav = NTF(mode="w+b",suffix=".wav")
	
	try:
		tts = gTTS(response_text,lang='en')
		tts.save(response_mp3.name)
	except:
		tts = gTTS("Hmm, hold on a sec.",lang='en')
		tts.save(response_mp3.name)
		
	subprocess.run(["ffmpeg", "-y", "-i", response_mp3.name, response_wav.name])
	subprocess.run(["sox", response_wav.name, temp_wav.name, "tempo", "1.2"])
	subprocess.run(["sox", temp_wav.name, response_wav.name, "bend", ".10,400,.25"])

	await do_avatar(response_text, response_wav, final_outfile)

	response_mp3.close()
	response_wav.close()


async def play_response(query, message):
	global IDLE

	while not IDLE:
		await asyncio.sleep(random.randint(0,10)/10)
	else:
		IDLE = False

	vote_button = Button(label="Vote!", url="https://top.gg/bot/808215286914351154", style=discord.ButtonStyle.blurple)
	waifu_ai_button = Button(label="Check out WaifuAI!", url="https://waifuai.com/", style=discord.ButtonStyle.blurple)
   
	view = View()
	view.add_item(vote_button)
	view.add_item(waifu_ai_button)
	response = waifu_query(query, str(message.author.id), str(message.author.name))

	final_outfile = NTF(mode="w+b",suffix=".mp4")

	await do_tts(response, final_outfile)

	try:
		await message.channel.send(file=discord.File(final_outfile.name), reference=message, view=view)
	except discord.errors.Forbidden:
		await message.channel.send(file=discord.File(final_outfile.name), view=view)

	IDLE = True

	final_outfile.close()

client.run(CONFIG["discord-token"])
