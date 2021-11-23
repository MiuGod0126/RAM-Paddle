'''
paddle对齐torch跑了168轮的权重
torch:valid acc of 98.683,Test Acc: 9863/10000 (98.63% - 1.37%)
paddle:valid acc of ?,Test Acc: 9839/10000 (98.39% - 1.61%) 些许差距,由于模型中有随机采样导致

'''
import torch
import paddle

# 加载权重
tw_path= 'ckpt0/ram_6_8x8_1_model_best.pth.tar'
pw_path= 'ckpt0/ram_6_8x8_1_model_best.pdparams'
torch_state=torch.load(tw_path)['model_state']
paddle_params=paddle.load(pw_path)
paddle_state=paddle_params['model_state']

# 打印torch参数名和shape
print('----------torch key----------')
# for k,v in torch_state.items():
#     print(k,list(v.shape))
'''
sensor.fc1.weight [128, 64]
sensor.fc1.bias [128]
sensor.fc2.weight [128, 2]
sensor.fc2.bias [128]
sensor.fc3.weight [256, 128]
sensor.fc3.bias [256]
sensor.fc4.weight [256, 128]
sensor.fc4.bias [256]
rnn.i2h.weight [256, 256]
rnn.i2h.bias [256]
rnn.h2h.weight [256, 256]
rnn.h2h.bias [256]
locator.fc.weight [128, 256]
locator.fc.bias [128]
locator.fc_lt.weight [2, 128]
locator.fc_lt.bias [2]
classifier.fc.weight [10, 256]
classifier.fc.bias [10]
baseliner.fc.weight [1, 256]
baseliner.fc.bias [1]
'''
# 打印paddle参数名字和对应的shape
print('----------paddle key----------')
# for k,v in paddle_state.items():
#     print(k,v.shape)
'''
sensor.fc1.weight [64, 128]
sensor.fc1.bias [128]
sensor.fc2.weight [2, 128]
sensor.fc2.bias [128]
sensor.fc3.weight [128, 256]
sensor.fc3.bias [256]
sensor.fc4.weight [128, 256]
sensor.fc4.bias [256]
rnn.i2h.weight [256, 256]
rnn.i2h.bias [256]
rnn.h2h.weight [256, 256]
rnn.h2h.bias [256]
locator.fc.weight [256, 128]
locator.fc.bias [128]
locator.fc_lt.weight [128, 2]
locator.fc_lt.bias [2]
classifier.fc.weight [256, 10]
classifier.fc.bias [10]
baseliner.fc.weight [256, 1]
baseliner.fc.bias [1]
'''


def align_weight(torch_state,paddle_state,save_path=None):
    # torch key转paddle key
    torch_to_paddle_keys = {}
    # 多余的torch key
    skip_keys = []
    # weight中二维缺不需要转置的
    donot_transpose = []


    for i,(tk, tw) in enumerate(torch_state.items()):
        transpose = False  # 仅打印
        # 跳过不需要的w
        if tk in skip_keys:
            continue
        # 转置linear的weight
        if tk.find('.weight') != -1:
            if tk not in donot_transpose:
                if tw.ndim == 2:
                    tw = tw.transpose(0, 1)
                    transpose = True
        # 转换key名
        pk = tk
        for k, v in torch_to_paddle_keys.items():
            pk = pk.replace(k, v)
        print(f"[{i+1}/{len(torch_state)}]Converting: {tk} => {pk} | is_transpose {transpose}")
        paddle_state[pk] = tw.cpu().detach().numpy()
    if save_path is not None:
        paddle.save(paddle_state,save_path)
    print('Align Over!')
    return paddle_state

# 权重对齐
# paddle_state=align_weight(torch_state,paddle_state,save_path=None)
# paddle_params['model_state']=paddle_state
# paddle.save(paddle_params, pw_path)

# 误差对比
import numpy as np
from model import RecurrentAttention
tout=np.load('tout.npy',allow_pickle=True)[0]
for k,v in tout.items():
    for i in range(len(v)):
        tout[k][i]=paddle.to_tensor(tout[k][i])
# print(tout)


@paddle.no_grad()
def test_diff(tout,model):
    paddle.set_device("cpu")
    paddle.set_grad_enabled(False)
    # model.eval()
    # torch 输入输出
    out_name=['h_t', 'l_t', 'b_t', 'log_probas', 'p']
    x, l_t,h_t  = tout['input'] # 顺序颠倒咯

    # forward
    num_glimpses = 6
    for t in range(num_glimpses - 1):
        # forward pass through model
        h_t, l_t, b_t, p = model(x, l_t, h_t)

    # last iteration
    paddle_out= model(x, l_t, h_t, last=True)
    # for i in range(len(paddle_out)):paddle_out[i]=paddle_out[i].numpy()
    for out_name,to,po in zip(out_name,tout['output'],paddle_out):
        diff1 = abs(to.numpy() - po.numpy()).mean()
        diff2 = abs(to.numpy() - po.numpy()).max()
        print(f'out={out_name},mean diff={diff1},max diff={diff2}')

# params
patch_size=8
num_patches=1
glimpse_scale=1
num_channels=1
loc_hidden=128
glimpse_hidden=128
std=0.05
hidden_size=256
num_classes=10
# model
model=RecurrentAttention(patch_size,num_patches,glimpse_scale,num_channels,
                        loc_hidden,glimpse_hidden,std,hidden_size,num_classes)

model.set_dict(paddle_state)
test_diff(tout,model)