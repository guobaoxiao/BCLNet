import torch
import torch.nn as nn
from loss import batch_episym
from einops import rearrange
import torch.nn.functional as F


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)


class DGCNN_Layer(nn.Module):
    def __init__(self, knn_num=10, in_channel=128):
        super(DGCNN_Layer, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 3), stride=(1, 3)),
                # [32,128,2000,9]→[32,128,2000,3]
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),  # [32,128,2000,3]→[32,128,2000,1]
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        if self.knn_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 3), stride=(1, 3)),
                # [32,128,2000,6]→[32,128,2000,2]
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)),  # [32,128,2000,2]→[32,128,2000,1]
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        # print(self.knn_num)
        out = self.conv(x)  # BCN1
        return out


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # inner[32,2000,2000]内积？
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # xx[32,1,2000]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # distance[32,2000,2000]****记得回头看

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k) [32,2000,9] [32,1000,6]

    return idx[:, :, :]


def get_graph_feature(x, k=20, idx=None):
    # x[32,128,2000,1],k=9
    # x[32,128,1000,1],k=6
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # x[32,128,2000]
    if idx is None:
        idx_out = knn(x, k=k)  # idx_out[32,2000,9]
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx_out + idx_base  # idx[32,2000,9] 把32个批次的标号连续了

    idx = idx.view(-1)  # idx[32*2000*9] 把32个批次连在一起了 [32*1000*6]

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # x[32,2000,128]
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)  # feature[32,2000,9,128]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # x[32,2000,9,128]
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()  # feature[32,256,2000,9] 图特征
    return feature


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BCMAttention(nn.Module):
    def __init__(self, channels, num_heads, k_num=20):
        super(BCMAttention, self).__init__()
        self.k_num = k_num
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # h11

        self.query_filter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 1)),
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU())
        self.key_filter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 1)),
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU())
        self.value_filter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 1)),
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU())
        self.gcn_q = DGCNN_Layer(knn_num=self.k_num, in_channel=channels)
        self.gcn_k = DGCNN_Layer(knn_num=self.k_num, in_channel=channels)
        self.gcn_v = DGCNN_Layer(knn_num=self.k_num, in_channel=channels)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=(1, 1))

    def forward(self, x):
        B, C, N, _ = x.shape
        q = self.query_filter(x)
        k = self.key_filter(x)
        v = self.value_filter(x)

        q = self.gcn_q(get_graph_feature(q, k=self.k_num))
        k = self.gcn_k(get_graph_feature(k, k=self.k_num))
        v = self.gcn_v(get_graph_feature(v, k=self.k_num))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=N, w=1)

        out = self.project_out(out)
        return out + x


class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                trans(1,2))
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1,2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class BEBlock(nn.Module):
    def __init__(self, channels, r=4, k_num=8):
        super(BEBlock, self).__init__()
        self.k_num = k_num
        inter_channels = int(channels // r)

        self.project_be = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            )

        self.project_knn = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # b*128*1*1
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att = DGCNN_Layer(self.k_num, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.project_be(x)

        x_local = self.project_knn(x1)
        x_local = self.local_att(get_graph_feature(x_local, k=self.k_num))

        xlg = self.global_att(x1) + x_local
        weight = self.sigmoid(xlg)
        return weight


class BCMA(nn.Module):
    def __init__(self, channels, num_heads=4, k_num=8):
        super(BCMA, self).__init__()
        self.k_num = k_num
        self.norm1 = nn.LayerNorm(channels)
        self.attn = BCMAttention(channels, num_heads, self.k_num)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = ResNet_Block(channels, channels, pre=False)

    def forward(self, x):
        b, c, h, w = x.shape
        xs = to_4d(self.norm1(to_3d(x)), h, w)
        x = x + self.attn(xs)
        xs = to_4d(self.norm2(to_3d(x)), h, w)
        x = x + self.ffn(xs)
        return x  # BCN1


class BCRBlock(nn.Module):
    def __init__(self, channels, k_num=8, r=4):
        super(BCRBlock, self).__init__()
        self.channels = channels
        self.ratio = r
        self.k = k_num
        self.Weight = BEBlock(channels, self.ratio, self.k)
        self.resnet_1 = ResNet_Block(self.channels * 2, self.channels, pre=True)
        self.resnet_2 = nn.Sequential(
            ResNet_Block(self.channels, self.channels, pre=False),
            ResNet_Block(self.channels, self.channels, pre=False),
            ResNet_Block(self.channels, self.channels, pre=False)
        )

    def forward(self, x):
        x_1 = self.resnet_1(x)
        wei = self.Weight(x_1)
        x = x_1 * wei
        x = self.resnet_2(x)
        return x


class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5, clusters=200):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),  # 4或6 → 128
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.down_1 = diff_pool(self.out_channel, clusters)
        self.l1 = []
        for _ in range(2):
            self.l1.append(OAFilter(self.out_channel, clusters))
        self.up_1 = diff_unpool(self.out_channel, clusters)
        self.l1 = nn.Sequential(*self.l1)

        self.down_2 = diff_pool(self.out_channel, clusters)
        self.l2 = []
        for _ in range(2):
            self.l2.append(OAFilter(self.out_channel, clusters))
        self.up_2 = diff_unpool(self.out_channel, clusters)
        self.l2 = nn.Sequential(*self.l2)

        self.embed_00 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.embed_01 = BCRBlock(self.out_channel, k_num=self.k_num)
        self.embed_02 = BCRBlock(self.out_channel, k_num=self.k_num)

        self.resfomer_1 = BCMA(self.out_channel, num_heads=4, k_num=self.k_num)
        self.resfomer_2 = BCMA(self.out_channel, num_heads=4, k_num=self.k_num)

        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False)
        )
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if self.predict:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)]  # indices[32,1000]剪枝剪掉一半
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)  # y_out 剪枝后保留的标签[32,1000]
            w_out = torch.gather(weights, dim=-1, index=indices)  # w_out 剪枝后保留的w0[32,1000]
        indices = indices.view(B, 1, -1, 1)  # indices[32,1,1000,1]

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))  # x_out剪枝后保留的x[32,1,1000,4]
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))  # x_out 剪枝后保留的x[32,1,500,4]
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))  # feature_out 剪枝后保留的features[32,128,500,1]
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        B, _, N, _ = x.size()
        out = x.transpose(1, 3).contiguous()  # contiguous断开out与x的依赖关系。out[32,4或6,2000,1]
        out = self.conv(out)  # out[32,128,2000,1]

        out = self.embed_00(out)  # out[32,128,2000,1] [32,128,1000,1]

        x1 = self.resfomer_1(out)

        x_down = self.down_1(out)
        x2 = self.l1(x_down)
        x_up = self.up_1(out, x2)

        out = self.embed_01(torch.cat([x1, x_up], dim=1))
        w0 = self.linear_0(out).view(B, -1)  # w0[32,2000]

        x3 = self.resfomer_2(out)
        x_down = self.down_2(out)
        x4 = self.l2(x_down)
        x_up = self.up_2(out, x4)

        out = self.embed_02(torch.cat([x3, x_up], dim=1))

        out = self.embed_1(out)
        w1 = self.linear_1(out).view(B, -1)  # w1[32,2000]

        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            out = self.embed_2(out)
            w2 = self.linear_2(out)
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat


class BCLNet(nn.Module):
    def __init__(self, config):
        super(BCLNet, self).__init__()

        self.ds_0 = DS_Block(initial=True, predict=False, out_channel=128, k_num=9,
                             sampling_rate=config.sr, clusters=config.clusters)  # sampling_rate=0.5
        self.ds_1 = DS_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr, clusters=config.clusters)

    def forward(self, x, y):
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0 = self.ds_0(x, y)  # 返回的是x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)  # 变成0到1的权重[32,1,1000,1]
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)  # x_[32,1,1000,6] 剪枝后的特征并带上了权重信息

        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1)  # x_[32,1,1000,6],y1[32,1000]

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)  # y_hat对称极线距离

        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    mask = logits[:, 0, :, 0]
    weights = logits[:, 1, :, 0]

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen

    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat




