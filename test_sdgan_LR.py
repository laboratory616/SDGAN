import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
import os
from torchvision.datasets import ImageFolder
import torchvision
import sys
from models.adgan import NCSNpp
import os
import torch.nn as nn
from torch.backends import cudnn
import time
from models.discriminator import Discriminator_small, Discriminator_large

sys.path.append('')
sys.path.append('')

from pytorch_fid.fid_score import calculate_fid_given_paths

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


# %% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                    (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, T, opt, latent, list_i):

    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):  # 采样步数，n_time=4
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = latent
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)  # 得到xt-1
            x = x_new.detach()
    return x


def sample_and_test(args):
    torch.manual_seed(42)
    device = 'cuda:0'
    batch_size = args.batch_size

    # 读取数据
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ImageFolder("", transform=data_transform)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(dataset,args.world_size)
    read_path = ''
    img_list = os.listdir(read_path)
    list_i = 0

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True
                                              # sampler=test_sampler,
                                              # drop_last=True
                                              )

    to_range_0_1 = lambda x: (x + 1.) / 2.
    netG = NCSNpp(args).to(device)

    ckpt = torch.load('./saved_info/sdgan/ddgan_{}/netG_{}.pth'.format(args.net_type, args.epoch_id),
                      map_location=device)

    cudnn.benchmark = True
    # loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()

    # T和参数
    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    # 保存路径
    save_dir = "./generated_samples/{}".format(args.dataset)
    # 833改为100
    start_time1 = np.zeros(100)
    end_time1 = np.zeros(100)
    run_time1 = np.zeros(100)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for iteration, (x, y) in enumerate(data_loader):
        print(list_i)
        # start_time1[list_i] = time.perf_counter()
        w = x.size(3)
        split_size = int(w // 2)

        left, right = x.split(split_size, 3)
        real_data = left.to(device, non_blocking=True)
        right_data = right.to(device, non_blocking=True)
        x_t_1 = right_data
        latent = right_data

        start_time1[list_i] = time.perf_counter()
        fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args, latent, list_i)
        end_time1[list_i] = time.perf_counter()
        run_time1[list_i] = end_time1[list_i] - start_time1[list_i]
        run_time1[list_i] = round(run_time1[list_i], 3)
        fake_sample = to_range_0_1(fake_sample)
        torchvision.utils.save_image(fake_sample,
                                     './generated_samples/{}/{}.png'.format(args.dataset,str(list_i).zfill(3)))
        # real_data = to_range_0_1(real_data)
        # torchvision.utils.save_image(real_data,
        #                              './generated_samples/Revise/20230920/GST/{}.png'.format(str(list_i).zfill(3)))
        # right_data = to_range_0_1(right_data)
        # torchvision.utils.save_image(right_data,
        #                              './generated_samples/Revise/20230920/HMI/{}.png'.format(str(list_i).zfill(3)))
        args.number = args.number + 1
        # end_time1[list_i] = time.perf_counter()
        # run_time1[list_i] = end_time1[list_i] - start_time1[list_i]
        # run_time1[list_i] = round(run_time1[list_i],3)
        #print(list_i, "时间差：", str(run_time1[list_i]))
        list_i = list_i + 1
    # print(str(run_time1), sep=", ")
    #print(str(run_time1))
    print(','.join([str(run_time1[i]) for i in range(100)]))


if __name__ == '__main__':
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                        help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int, default=1000)
    parser.add_argument('--net_type', default='z')
    parser.add_argument('--number', type=int, default=0)
    parser.add_argument('--sum_sample', type=int, default=4)
    parser.add_argument('--num_channels', type=int, default=3,
                        help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--num_channels_dae', type=int, default=64,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # genrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy',
                        help='directory to real canchasagan for FID computation')

    parser.add_argument('--dataset', default='sun_z', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size of image')
    parser.add_argument('--nz', type=int, default=196608)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='sample generating batch size')
    parser.add_argument('--world_size', type=int, default=1)

    args = parser.parse_args()
    sample_and_test(args)
    end_time = time.perf_counter()
    run_time = end_time - start_time  # 程序的运行时间，单位为秒
    run_time = round(run_time,3)
    print("采样总时间：", str(run_time))

