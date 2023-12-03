import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import requests

def new_filename(path, filename) -> str:
    i = 1
    while len(glob.glob(f"{path}/{filename}-{i}.*")) > 0:
        i += 1
    return f'{path}/{filename}-{i}'

def new_result_path(path, foldername) -> str:
    i = 1
    while os.path.exists(f'{path}/{foldername}-{i}'):
        i += 1
    return f'{path}/{foldername}-{i}', f'{foldername}-{i}'


def rt_plot_init(time_arr, ept_arr, papr, mse=None, bl=False, action_div=None):
    lines, = plt.plot(time_arr, ept_arr)
    if bl is True:
        if action_div is None:
            plot_text_bl = plt.figtext(0.02, 0.02, f'0 Best PAPR: {papr:.04f} dB', ha='left', color='red')
        else:
            plt.subplots_adjust(bottom=10/72)
            plot_text_bl = plt.figtext(0.02, 0.02, f'0 Best PAPR: {papr:.04f} dB / action_div: x{action_div}', ha='left', color='red')
    else:
        plot_text_bl = None
    if mse is None:
        plot_text_br = plt.figtext(0.98, 0.02, f'PAPR: {papr:.04f} dB', ha='right', color='red')
    else:
        plot_text_br = plt.figtext(0.98, 0.02, f'PAPR: {papr:.04f} dB / All_MSE: {mse:.04f}', ha='right', color='red')
    plt.xlabel('Time')
    plt.xlim(0, 1)
    plt.xticks([0, 0.5, 1], [0, 'T/2', 'T'])
    plt.ylabel('EP(t)')
    plt.ylim(0, )
    plt.legend()
    plt.grid(True)
    return lines, plot_text_bl, plot_text_br

def rt_plot_reload_line(lines, time_arr, ept_arr, setcolor='gray'):
    lines.set_data(time_arr, ept_arr)
    lines.set_color(setcolor)

def rt_plot_reload_text_bl(text, index, best_papr, action_div=None, setcolor='gray'):
    if action_div is None:
        text.set_text(f'{index} Best PAPR: {best_papr:.03f} dB')
    else:
        text.set_text(f'{index+1} Best PAPR: {best_papr:.03f} dB / action_div: x{action_div:.06f}')
    text.set_color(setcolor)

def rt_plot_reload_text_br(text, papr, mse=None, setcolor='gray'):
    if mse is None:
        text.set_text(f'PAPR: {papr:.03f} dB')
    else:
        text.set_text(f'PAPR: {papr:.03f} dB / All_MSE: {mse:.03f}')
    text.set_color(setcolor)

def rt_pause_plot():
    plt.pause(.01)

def close_plot():
    plt.clf()
    plt.close()

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def send_line(channel_token, user_id, text):
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {channel_token}'}
    post = {'to': user_id, 'messages': [{'type': 'text', 'text': text}]}
    req = requests.post(url, headers=headers, data=json.dumps(post))

    if req.status_code != 200:
        print(req.text)
