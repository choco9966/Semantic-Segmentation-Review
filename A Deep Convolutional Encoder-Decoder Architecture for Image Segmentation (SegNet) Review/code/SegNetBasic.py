import torch
import torch.nn as nn
class SegNetBasic(nn.Module):
    def __init__(self, num_classes=21):
        super(SegNetBasic, self).__init__()
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)
            return cbr

        def CB(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]

            cb = nn.Sequential(*layers)
            return cb
        
        # conv1 
        self.cbr1_1 = CBR(3, 64, 7, 1, 3)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True) 

        # conv2 
        self.cbr2_1 = CBR(64, 64, 7, 1, 3)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True) 

        # conv3
        self.cbr3_1 = CBR(64, 64, 7, 1, 3)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True) 

        # conv4
        self.cbr4_1 = CBR(64, 64, 7, 1, 3)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True) 

        # Deconv4
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.dcb4_1 = CB(64, 64, 7, 1, 3)

        # Deconv3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.dcb3_1 = CB(64, 64, 7, 1, 3)
        
        # Deconv2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.dcb2_1 = CB(64, 64, 7, 1, 3)
        
        # Deconv1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.dcb1_1 = CB(64, 64, 7, 1, 3)
        self.score_fr = nn.Conv2d(64, num_classes, kernel_size = 1, bias=False)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

                # xavier_uniform은 bias에 대해서는 제공하지 않음 
                # ValueError: Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        h = self.cbr1_1(x)
        dim1 = h.size()
        h, pool1_indices = self.pool1(h)
        
        h = self.cbr2_1(h)
        dim2 = h.size()
        h, pool2_indices = self.pool2(h)
        
        h = self.cbr3_1(h)
        dim3 = h.size()
        h, pool3_indices = self.pool3(h)
        
        h = self.cbr4_1(h)
        dim4 = h.size()
        h, pool4_indices = self.pool4(h)

        h = self.unpool4(h, pool4_indices, output_size = dim4)
        h = self.dcb4_1(h)
        
        h = self.unpool3(h, pool3_indices, output_size = dim3)
        h = self.dcb3_1(h)
        
        h = self.unpool2(h, pool2_indices, output_size = dim2)
        h = self.dcb2_1(h)
        
        h = self.unpool1(h, pool1_indices, output_size = dim1)
        h = self.dcb1_1(h)
        h = self.score_fr(h)        
        return torch.sigmoid(h)