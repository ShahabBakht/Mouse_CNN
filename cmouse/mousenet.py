import torch
from torch import nn
import networkx as nx
import numpy as np
from config import  INPUT_SIZE, EDGE_Z, OUTPUT_AREAS, HIDDEN_LINEAR, NUM_CLASSES
from change_net_config import *

class Conv2dMask(nn.Conv2d):
    """
    Conv2d with Gaussian mask 
    """
    def __init__(self, in_channels, out_channels, kernel_size, gsh, gsw, mask=3, stride=1, padding=0):
        super(Conv2dMask, self).__init__(in_channels, out_channels, kernel_size, stride=stride)
        self.mypadding = nn.ConstantPad2d(padding, 0)
        if mask == 0:
            self.mask = None
        if mask==1:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw)))
        elif mask ==2:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw)), requires_grad=False) 
        elif mask ==3:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask_vary_channel(gsh, gsw, kernel_size, out_channels, in_channels)), requires_grad=False)
        else:
            assert("mask should be 0, 1, 2, 3!")

    def forward(self, input):
        if self.mask is not None:
            return super(Conv2dMask, self).conv2d_forward(self.mypadding(input), self.weight*self.mask)
        else:
            return super(Conv2dMask, self).conv2d_forward(self.mypadding(input), self.weight)
            
    def make_gaussian_kernel_mask(self, peak, sigma):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        width = int(sigma*EDGE_Z)        
        x = np.arange(-width, width+1)
        X, Y = np.meshgrid(x, x)
        radius = np.sqrt(X**2 + Y**2)

        probability = peak * np.exp(-radius**2/2/sigma**2)

        re = np.random.rand(len(x), len(x)) < probability
        # plt.imshow(re, cmap='Greys')
        return re
    
    def make_gaussian_kernel_mask_vary_channel(self, peak, sigma, kernel_size, out_channels, in_channels):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :param kernel_size: kernel size of the conv2d 
        :param out_channels: number of output channels of the conv2d
        :param in_channels: number of input channels of the con2d
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        re = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        for i in range(out_channels):
            for j in range(in_channels):
                re[i, j, :] = self.make_gaussian_kernel_mask(peak, sigma)
        return re

class Conv3dMask(nn.Conv3d):
    """
    Conv3d with Gaussian mask 
    """
    def __init__(self, in_channels, out_channels, kernel_size, gsh, gsw, mask=3, stride=1, padding=0):
        super(Conv3dMask, self).__init__(in_channels, out_channels, kernel_size, stride=stride)
        self.mypadding = nn.ConstantPad3d(padding, 0)
        if mask == 0:
            self.mask = None
        if mask==1:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask_3d(gsh, gsw)))
        elif mask ==2:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask_3d(gsh, gsw)), requires_grad=False) 
        elif mask ==3:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask_vary_channel_3d(gsh, gsw, kernel_size, out_channels, in_channels)), requires_grad=False)
        else:
            assert("mask should be 0, 1, 2, 3!")
        self.weight = nn.Parameter(self.weight*self.mask)

    def forward(self,input):
        
        return super(Conv3dMask, self).forward(self.mypadding(input))

    def make_gaussian_kernel_mask_3d(self, peak, sigma):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        width = int(sigma*EDGE_Z)        
        x = np.arange(-width, width+1)
        z = np.arange(-1,2)
        Z, X, Y = np.meshgrid(x,z,x)
        radius = np.sqrt(Z**2 + X**2 + Y**2)

        probability = peak * np.exp(-radius**2/2/sigma**2)

        re = np.random.rand(len(z), len(x), len(x)) < probability
        # plt.imshow(re, cmap='Greys')
        return re
    
    def make_gaussian_kernel_mask_vary_channel_3d(self, peak, sigma, kernel_size, out_channels, in_channels):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :param kernel_size: kernel size of the conv2d 
        :param out_channels: number of output channels of the conv2d
        :param in_channels: number of input channels of the con2d
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        re = np.zeros((out_channels, in_channels, *kernel_size))
        for i in range(out_channels):
            for j in range(in_channels):
                # print(self.make_gaussian_kernel_mask_3d(peak, sigma).shape)
                re[i, j, :] = self.make_gaussian_kernel_mask_3d(peak, sigma)
        return re

class ConvGRUCellMask(nn.Module):
    ''' Initialize ConvGRU cell '''
    def __init__(self, input_size, hidden_size, kernel_size, gsh, gsw, padding, stride=1):
        super(ConvGRUCellMask, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        padding = kernel_size // 2
        self.padding = padding

        self.reset_gate = Conv2dMask(input_size+hidden_size, hidden_size, kernel_size, gsh=gsh, gsw=gsw, mask=3, stride=stride, padding=padding)
        self.update_gate = Conv2dMask(input_size+hidden_size, hidden_size, kernel_size, gsh=gsh, gsw=gsw, mask=3, stride=stride, padding=padding)
        self.out_gate = Conv2dMask(input_size+hidden_size, hidden_size, kernel_size, gsh=gsh, gsw=gsw, mask=3, stride=stride, padding=padding)

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.update_gate.bias, 0.)
        nn.init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_tensor, hidden_state):
        if hidden_state is None:
            B, C, *spatial_dim = input_tensor.size()
            if torch.cuda.is_available():
                hidden_state = torch.zeros([B,self.hidden_size,*spatial_dim]).cuda()
            else:
                hidden_state = torch.zeros([B,self.hidden_size,*spatial_dim])
        # [B, C, H, W]
        combined = torch.cat([input_tensor, hidden_state], dim=1) #concat in C
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        # print(self.stride,self.padding)
        # print(input_tensor.shape,update.shape)
        out = torch.tanh(self.out_gate(torch.cat([input_tensor, hidden_state * reset], dim=1)))
        
        new_state = hidden_state * (1 - update) + out * update
        return new_state


class ConvGRUMask(nn.Module):
    ''' Initialize a multi-layer Conv GRU '''
    def __init__(self, input_size, hidden_size, kernel_size, gsh, gsw, num_layers, padding, dropout=0.1, stride=1):
        super(ConvGRUMask, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.gsh = gsh
        self.gsw = gsw
        self.stride = 1
        self.padding = padding

        cell_list = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_size
            cell = ConvGRUCellMask(input_dim, self.hidden_size, self.kernel_size, self.gsh, self.gsw, self.padding, self.stride)
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cell_list.append(getattr(self, name))
        
        self.cell_list = nn.ModuleList(cell_list)
        self.dropout_layer = nn.Dropout(p=dropout)


    def forward(self, x, hidden_state=None):
        [B, seq_len, *_] = x.size()

        if hidden_state is None:
            hidden_state = [None] * self.num_layers
        # input: image sequences [B, T, C, H, W]
        current_layer_input = x 
        del x

        last_state_list = []

        for idx in range(self.num_layers):
            cell_hidden = hidden_state[idx]
            output_inner = []
            for t in range(seq_len):
                cell_hidden = self.cell_list[idx](current_layer_input[:,t,:], cell_hidden)
                cell_hidden = self.dropout_layer(cell_hidden) # dropout in each time step
                output_inner.append(cell_hidden)

            layer_output = torch.stack(output_inner, dim=1)
            current_layer_input = layer_output

            last_state_list.append(cell_hidden)

        last_state_list = torch.stack(last_state_list, dim=1)

        return layer_output#, last_state_list

class MouseNet(nn.Module):
    """
    torch model constructed by parameters provided in network.
    """
    def __init__(self, network, mask=3, bn=1):
        super(MouseNet, self).__init__()
        self.Convs = nn.ModuleDict()
        self.bn = bn
        if self.bn:
            self.BNs = nn.ModuleDict()
        network = change_net_config(network)
        self.network = network
        
        G, _ = network.make_graph()
        Gtop = nx.topological_sort(G)
        root = next(Gtop) # get root of graph
        self.edge_bfs = [e for e in nx.edge_bfs(G, root)] # traversal edges by bfs
        
        for e in self.edge_bfs:
            layer = network.find_conv_source_target(e[0], e[1])
            params = layer.params   

            self.Convs[e[0]+e[1]] = Conv2dMask(params.in_channels, params.out_channels, params.kernel_size,
                                               params.gsh, params.gsw, stride=params.stride, mask=mask, padding=params.padding)
            ## plotting Gaussian mask
            #plt.title('%s_%s_%sx%s'%(e[0].replace('/',''), e[1].replace('/',''), params.kernel_size, params.kernel_size))
            #plt.savefig('%s_%s'%(e[0].replace('/',''), e[1].replace('/','')))
            if self.bn:
                self.BNs[e[0]+e[1]] = nn.BatchNorm2d(params.out_channels)

        # calculate total size output to classifier
        total_size=0
        for area in OUTPUT_AREAS:
            if area =='VISp5':
                layer = network.find_conv_source_target('VISp2/3','VISp5')
                visp_out = layer.params.out_channels
                # create 1x1 Conv downsampler for VISp5
                visp_downsample_channels = 32
                ds_stride = 2
                self.visp5_downsampler = nn.Conv2d(visp_out, visp_downsample_channels, 1, stride=ds_stride)
                total_size += INPUT_SIZE[1]/ds_stride * INPUT_SIZE[2]/ds_stride * visp_downsample_channels
            else:
                layer = network.find_conv_source_target('%s2/3'%area[:-1],'%s'%area)
                total_size += int(layer.out_size*layer.out_size*layer.params.out_channels)
        
        self.classifier = nn.Sequential(
            nn.Linear(int(total_size), HIDDEN_LINEAR),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(HIDDEN_LINEAR, HIDDEN_LINEAR),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(HIDDEN_LINEAR, NUM_CLASSES),
        )

    def get_img_feature(self, x, area_list):
        """
        function for get activations from a list of layers for input x
        :param x: input image set Tensor with size (num_img, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
        :param area_list: a list of area names
        :return: if list length is 1, return the flattened activation of that area; 
                 if list length is >1, return concatenated flattened activation of the areas.
        """
        calc_graph = {}
        for e in self.edge_bfs:
            if e[0] == 'input':
                if self.bn:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](x)))
                else:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](x))
            else:
                if e[1] in calc_graph:
                    if self.bn:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]])))
                    else:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](calc_graph[e[0]]))
                else:
                    if self.bn:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]])))
                    else:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](calc_graph[e[0]]))
        
        if len(area_list) == 1:
            return torch.flatten(calc_graph['%s'%(area_list[0])], 1)
        else:
            re = None
            for area in area_list:
                if area == 'VISp5':
                    re=torch.flatten(self.visp5_downsampler(calc_graph['VISp5']), 1)
                else:
                    if re is not None:
                        re = torch.cat([torch.flatten(calc_graph[area], 1), re], axis=1)
                    else:
                        re = torch.flatten(calc_graph[area], 1)
        return re

    def get_img_feature_no_flatten(self, x, area_list):
        """
        function for get activations from a list of layers for input x
        :param x: input image set Tensor with size (BN, C, SL, H, W) # following CPC standard
        :param area_list: a list of area names
        :return: if list length is 1, return the activation of that area; 
                 if list length is >1, return concatenated activation of the areas along the channel axis.
        """
        #area_list = self.param['output_area_list']
        calc_graph = {}
        for e in self.edge_bfs:
            if e[0] == 'input':
                if self.bn:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](x)))
                else:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](x))
            else:
                if e[1] in calc_graph:
                    if self.bn:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]])))
                    else:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](calc_graph[e[0]]))
                else:
                    if self.bn:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]])))
                    else:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](calc_graph[e[0]]))
        if len(area_list) == 1:
            return calc_graph['%s'%(area_list[0])]
        else:
            re = None
            for area in area_list:
                if area == 'VISp5':
                    re=self.visp5_downsampler(calc_graph['VISp5'])
                else:
                    if re is not None:
                        re = torch.cat([calc_graph[area], re], axis=1)
                    else:
                        re = calc_graph[area]
        return re
    def forward(self, x):
        ''' 
        input x shape follows CPC standards (BN, C, SL, H, W)
        B: batch size, N: number of blocks, C: number of channels, SL: block size, H,W: image size
        ''' 
        (BN, C, SL, H, W) = x.shape
        x = x.permute(0,2,1,3,4).contiguous().view((BN*SL,C,H,W))   
        x = self.get_img_feature_no_flatten(x, OUTPUT_AREAS)
        (T,C,H,W) = x.shape
        x = x.view(BN,SL,C,H,W).permute(0,2,1,3,4)
#         x = self.classifier(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)

class MouseNetGRU(nn.Module):
    """
    torch model constructed by parameters provided in network.
    """
    def __init__(self, network, mask=3, bn=1):
        super(MouseNetGRU, self).__init__()
        self.Convs = nn.ModuleDict()
        self.bn = bn
        if self.bn:
            self.BNs = nn.ModuleDict()
        self.network = network
        
        G, _ = network.make_graph()
        Gtop = nx.topological_sort(G)
        root = next(Gtop) # get root of graph
        self.edge_bfs = [e for e in nx.edge_bfs(G, root)] # traversal edges by bfs
        
        for e in self.edge_bfs:
            layer = network.find_conv_source_target(e[0], e[1])
            params = layer.params   

            self.Convs[e[0]+e[1]] = ConvGRUMask(params.in_channels, params.out_channels, params.kernel_size,
                                               params.gsh, params.gsw, num_layers = 1, stride=params.stride, padding=params.padding)
            ## plotting Gaussian mask
            #plt.title('%s_%s_%sx%s'%(e[0].replace('/',''), e[1].replace('/',''), params.kernel_size, params.kernel_size))
            #plt.savefig('%s_%s'%(e[0].replace('/',''), e[1].replace('/','')))
            if self.bn:
                self.BNs[e[0]+e[1]] = nn.BatchNorm3d(params.out_channels)

        # calculate total size output to classifier
        total_size=0
        for area in OUTPUT_AREAS:
            if area =='VISp5':
                layer = network.find_conv_source_target('VISp2/3','VISp5')
                visp_out = layer.params.out_channels
                # create 1x1 Conv downsampler for VISp5
                visp_downsample_channels = 32
                ds_stride = 2
                self.visp5_downsampler = nn.Conv2d(visp_out, visp_downsample_channels, 1, stride=ds_stride)
                total_size += INPUT_SIZE[1]/ds_stride * INPUT_SIZE[2]/ds_stride * visp_downsample_channels
            else:
                layer = network.find_conv_source_target('%s2/3'%area[:-1],'%s'%area)
                total_size += int(layer.out_size*layer.out_size*layer.params.out_channels)
        
        self.classifier = nn.Sequential(
            nn.Linear(int(total_size), HIDDEN_LINEAR),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(HIDDEN_LINEAR, HIDDEN_LINEAR),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(HIDDEN_LINEAR, NUM_CLASSES),
        )

    def get_img_feature(self, x, area_list):
        """
        function for get activations from a list of layers for input x
        :param x: input image set Tensor with size (num_img, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
        :param area_list: a list of area names
        :return: if list length is 1, return the flattened activation of that area; 
                 if list length is >1, return concatenated flattened activation of the areas.
        """
        calc_graph = {}
        for e in self.edge_bfs:
            if e[0] == 'input':
                if self.bn:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](x)))
                else:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](x))
            else:
                if e[1] in calc_graph:
                    if self.bn:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]])))
                    else:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](calc_graph[e[0]]))
                else:
                    if self.bn:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]])))
                    else:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](calc_graph[e[0]]))
        
        if len(area_list) == 1:
            return torch.flatten(calc_graph['%s'%(area_list[0])], 1)
        else:
            re = None
            for area in area_list:
                if area == 'VISp5':
                    re=torch.flatten(self.visp5_downsampler(calc_graph['VISp5']), 1)
                else:
                    if re is not None:
                        re = torch.cat([torch.flatten(calc_graph[area], 1), re], axis=1)
                    else:
                        re = torch.flatten(calc_graph[area], 1)
        return re

    def get_img_feature_no_flatten(self, x, area_list):
        """
        function for get activations from a list of layers for input x
        :param x: input image set Tensor with size (BN, C, SL, H, W) # following CPC standard
        :param area_list: a list of area names
        :return: if list length is 1, return the activation of that area; 
                 if list length is >1, return concatenated activation of the areas along the channel axis.
        """
        #area_list = self.param['output_area_list']
        calc_graph = {}
        for e in self.edge_bfs:
            if e[0] == 'input':
                if self.bn:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](x).permute(0,2,1,3,4))).permute(0,2,1,3,4)
                else:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](x))
            else:
                if e[1] in calc_graph:

                    if ((e[0] == 'VISp2/3') and (e[1] != 'VISp5')) or (e[0] == 'VISp5'):
                        B,T,C,W,H = calc_graph[e[0]].shape
                        temp_input = nn.functional.max_pool2d(calc_graph[e[0]].reshape((B*T,C,W,H)).contiguous(),kernel_size = 2, stride = 2).reshape((B,T,C,int(W/2),int(H/2))).contiguous()
                        
                    else:
                        temp_input = calc_graph[e[0]]

                    if self.bn:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](temp_input).permute(0,2,1,3,4))).permute(0,2,1,3,4)
                    else:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](temp_input))

                else:

                    if ((e[0] == 'VISp2/3') and (e[1] != 'VISp5')) or (e[0] == 'VISp5'):
                        B,T,C,W,H = calc_graph[e[0]].shape
                        temp_input = nn.functional.max_pool2d(calc_graph[e[0]].reshape((B*T,C,W,H)).contiguous(),kernel_size = 2, stride = 2).reshape((B,T,C,int(W/2),int(H/2))).contiguous()
                        
                    else:
                        temp_input = calc_graph[e[0]]

                    if self.bn:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](temp_input).permute(0,2,1,3,4))).permute(0,2,1,3,4)
                    else:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](temp_input))

        if len(area_list) == 1:
            return calc_graph['%s'%(area_list[0])]
        else:
            re = None
            for area in area_list:
                if area == 'VISp5':
                    B, T, C, W, H = calc_graph['VISp5'].shape
                    re=self.visp5_downsampler(calc_graph['VISp5'].reshape((B*T,C,W,H)).contiguous()).reshape((B,T,32,W//2,H//2)).contiguous()

                else:
                    if re is not None:
                        print(calc_graph[area].shape,re.shape)
                        re = torch.cat([calc_graph[area], re], axis=2)
                    else:
                        re = calc_graph[area]
        return re
    def forward(self, x):
        ''' 
        input x shape follows CPC standards (BN, C, SL, H, W)
        B: batch size, N: number of blocks, C: number of channels, SL: block size, H,W: image size
        ''' 
        # (BN, C, SL, H, W) = x.shape
        x = x.permute(0,2,1,3,4).contiguous()#.view((BN*SL,C,H,W))   
        x = self.get_img_feature_no_flatten(x, OUTPUT_AREAS)
        # (T,C,H,W) = x.shape
        x = x.permute(0,2,1,3,4)
#         x = self.classifier(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)

class MouseNet_3d(nn.Module):
    """
    torch model constructed by parameters provided in network.
    """
    def __init__(self, network, mask=3, bn=1):
        super(MouseNet_3d, self).__init__()
        self.Convs = nn.ModuleDict()
        self.bn = bn
        if self.bn:
            self.BNs = nn.ModuleDict()
        self.network = network
        
        G, _ = network.make_graph()
        Gtop = nx.topological_sort(G)
        root = next(Gtop) # get root of graph
        self.edge_bfs = [e for e in nx.edge_bfs(G, root)] # traversal edges by bfs
        temporal_kernel_size = 3
        temporal_padding = 1

        for e in self.edge_bfs:
            layer = network.find_conv_source_target(e[0], e[1])
            params = layer.params   
            if type(params.padding) is tuple:
                padding_size = (*params.padding,temporal_padding,temporal_padding)
            else:
                padding_size = (params.padding,params.padding,params.padding,params.padding,temporal_padding,temporal_padding)

            stride_size = (1,params.stride,params.stride)

            self.Convs[e[0]+e[1]] = Conv3dMask(params.in_channels, params.out_channels, (temporal_kernel_size,params.kernel_size,params.kernel_size),
                                               params.gsh, params.gsw, stride=stride_size, mask=mask, 
                                               padding=padding_size)
            ## plotting Gaussian mask
            #plt.title('%s_%s_%sx%s'%(e[0].replace('/',''), e[1].replace('/',''), params.kernel_size, params.kernel_size))
            #plt.savefig('%s_%s'%(e[0].replace('/',''), e[1].replace('/','')))
            if self.bn:
                self.BNs[e[0]+e[1]] = nn.BatchNorm3d(params.out_channels)

        # calculate total size output to classifier
        total_size=0
        for area in OUTPUT_AREAS:
            if area =='VISp5':
                layer = network.find_conv_source_target('VISp2/3','VISp5')
                visp_out = layer.params.out_channels
                # create 1x1 Conv downsampler for VISp5
                visp_downsample_channels = 32
                ds_stride_w = 2
                ds_stride_h = 2
                ds_stride_t = 1
                self.visp5_downsampler = nn.Conv3d(visp_out, visp_downsample_channels, 1, stride=(ds_stride_t,ds_stride_w,ds_stride_h))
                total_size += INPUT_SIZE[1]/ds_stride_w * INPUT_SIZE[2]/ds_stride_h * visp_downsample_channels
            else:
                layer = network.find_conv_source_target('%s2/3'%area[:-1],'%s'%area)
                total_size += int(layer.out_size*layer.out_size*layer.params.out_channels)
        
        self.classifier = nn.Sequential(
            nn.Linear(int(total_size), HIDDEN_LINEAR),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(HIDDEN_LINEAR, HIDDEN_LINEAR),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(HIDDEN_LINEAR, NUM_CLASSES),
        )

    def get_img_feature(self, x, area_list):
        """
        function for get activations from a list of layers for input x
        :param x: input image set Tensor with size (num_img, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
        :param area_list: a list of area names
        :return: if list length is 1, return the flattened activation of that area; 
                 if list length is >1, return concatenated flattened activation of the areas.
        """
        calc_graph = {}
        for e in self.edge_bfs:
            if e[0] == 'input':
                if self.bn:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](x)))
                else:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](x))
            else:
                if e[1] in calc_graph:
                    if self.bn:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]])))
                    else:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](calc_graph[e[0]]))
                else:
                    if self.bn:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]])))
                    else:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](calc_graph[e[0]]))
        
        if len(area_list) == 1:
            return torch.flatten(calc_graph['%s'%(area_list[0])], 1)
        else:
            re = None
            for area in area_list:
                if area == 'VISp5':
                    re=torch.flatten(self.visp5_downsampler(calc_graph['VISp5']), 1)
                else:
                    if re is not None:
                        re = torch.cat([torch.flatten(calc_graph[area], 1), re], axis=1)
                    else:
                        re = torch.flatten(calc_graph[area], 1)
        return re

    def get_img_feature_no_flatten(self, x, area_list):
        """
        function for get activations from a list of layers for input x
        :param x: input image set Tensor with size (BN, C, SL, H, W) # following CPC standard
        :param area_list: a list of area names
        :return: if list length is 1, return the activation of that area; 
                 if list length is >1, return concatenated activation of the areas along the channel axis.
        """
        #area_list = self.param['output_area_list']
        calc_graph = {}
        for e in self.edge_bfs:
            if e[0] == 'input':
                if self.bn:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](x)))
                else:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](x))
            else:
                if e[1] in calc_graph:
                    if self.bn:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]])))
                    else:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](calc_graph[e[0]]))
                else:
                    if self.bn:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]])))
                    else:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](calc_graph[e[0]]))
        if len(area_list) == 1:
            return calc_graph['%s'%(area_list[0])]
        else:
            re = None
            for area in area_list:
                if area == 'VISp5':
                    re=self.visp5_downsampler(calc_graph['VISp5'])
                else:
                    if re is not None:
                        re = torch.cat([calc_graph[area], re], axis=1)
                    else:
                        re = calc_graph[area]
        return re
    def forward(self, x):
        ''' 
        input x shape follows CPC standards (BN, C, SL, H, W)
        B: batch size, N: number of blocks, C: number of channels, SL: block size, H,W: image size
        ''' 
        (BN, C, SL, H, W) = x.shape
        x = self.get_img_feature_no_flatten(x, OUTPUT_AREAS)

        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)


if __name__ == "__main__":

    # G = Conv3dMask(in_channels = 3, out_channels = 3, kernel_size = (5,15,15), gsh = 2, gsw = 7,padding=(0,0,0,0,2,2))
    # import matplotlib.pyplot as plt
    # plt.imshow(G.mask[0,0,:,:,1])
    # plt.savefig('test.png')

    x = torch.rand((40, 3, 5, 64, 64))
    import time
    import network
    net = network.load_network_from_pickle('../example/network_(3,64,64).pkl')
    tic = time.time()
    # model = MouseNet_3d(net)  
    model = MouseNet(net)    
    # model = MouseNetGRU(net)    
    out = model(x)
    print(time.time() - tic)
    import ipdb; ipdb.set_trace()
