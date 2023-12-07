import os
import glob
import numpy as np
import matplotlib.pyplot as plt
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
    plt.figure()
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

def rt_circle_init(theta_k_bins_diffs):
    plt.figure()
    # 値が 0~1 までの円を作るために、角度の配列を作成
    angles = np.linspace(0, 2*np.pi, 100)

    # 円の x 座標と y 座標を計算
    x = np.cos(angles)
    y = np.sin(angles)

    # 円をプロット
    plt.axis('equal')
    plt.plot(x, y, color='gray')

    # 円の中心から円周上の入力値に向かって線を引くために、入力値に対応する角度を計算
    const_lines = [i * 0.1 * 2*np.pi for i in range(10)]

    # 入力値に対応する円周上の点の x 座標と y 座標を計算
    const_lines_x = [np.cos(angle) for angle in const_lines]
    const_lines_y = [np.sin(angle) for angle in const_lines]

    # 円の中心から円周上の入力値に向かって線を引く
    for i in range(10):
        plt.plot([0, const_lines_x[i]], [0, const_lines_y[i]], color='gray', lw=0.5)

    # 円の中心から円周上の入力値に向かって線を引くために、入力値に対応する角度を計算
    input_angles = [i * 2*np.pi for i in theta_k_bins_diffs]

    # 入力値に対応する円周上の点の x 座標と y 座標を計算
    input_x = [np.cos(angle) for angle in input_angles]
    input_y = [np.sin(angle) for angle in input_angles]

    # 円の中心から円周上の入力値に向かって線を引く
    circle_lines = []
    for i in range(len(theta_k_bins_diffs)):
        temp_line, = plt.plot([0, input_x[i]], [0, input_y[i]], lw=0.75)
        circle_lines.append(temp_line)

    # グラフの表示範囲を設定
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    return circle_lines

def rt_circle_reload_line(circle_lines, theta_k_bins_diffs):
    input_angles = [i * 2*np.pi for i in theta_k_bins_diffs]
    input_x = [np.cos(angle) for angle in input_angles]
    input_y = [np.sin(angle) for angle in input_angles]

    for i in range(len(theta_k_bins_diffs)):
        # lineオブジェクトのデータを更新
        circle_lines[i].set_data([0, input_x[i]], [0, input_y[i]])

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
