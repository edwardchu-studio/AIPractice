#
# class dGenerator(nn.Module):
#     def __init__(self):
#         super(dGenerator,self).__init__()
#
#         self.g_dconv1=nn.ConvTranspose2d(4,8,5,2,2,0)
#         self.g_dconv2=nn.ConvTranspose2d(8,16,3,2,1,1)
#
#         self.z_dconv1=nn.ConvTranspose2d(8,16,5,2,2,0)
#         self.z_dconv2=nn.ConvTranspose2d(24,32,3,2,1,1)
#
#         self.z_dconv3=nn.ConvTranspose2d(32,1,stride=2,kernel_size=1,padding=10,output_padding=1)
#
#
#         self.m_fc=nn.Linear(32*32*32,56*56*1)
#         self.bn_g=nn.modules.BatchNorm2d(8)
#         self.bn_z=nn.modules.BatchNorm2d(12)
#         self.use_gpu=torch.cuda.is_available()
#
#     def forward(self, g,z):
#
#         if self.use_gpu:
#             g=g.view(-1,4,10,10).cuda()
#             # print(g.shape,z.shape)
#             z=z.cuda()
#             gdc1=F.relu(self.g_dconv1(g))
#             # print('gdc1.shape:',gdc1.shape)
#             # gc1=self.bn_g(gdc1)
#
#             gdc2=F.relu(self.g_dconv2(gdc1))
#             # print('gdc2.shape:',gdc2.shape)
#
#
#             zdc1 = F.relu(self.z_dconv1(z))
#             # print('zdc1.shape',zdc1.shape)
#
#             m1 = torch.cat([gdc1, zdc1], 1)
#
#             # print('m1.shape',m1.shape)
#             zdc2=F.relu(self.z_dconv2(m1))
#
#             # print('zdc2.shape',zdc2.shape)
#
#             # m2= torch.cat([gdc2, zdc2], 1)
#             # print('m2.shape',m2.shape)
#
#             # m2_r=zdc2.view(-1,32,32,32).cuda()
#             o = F.relu(self.z_dconv3(zdc2))
#             # print('output.shape',o.shape)
#             return o.cuda()
#         else:
#             g = g.view(-1, 4, 10, 10)
#             print(g.shape, z.shape)
#
#             gdc1 = F.relu(self.g_dconv1(g))
#             print('gdc1.shape:', gdc1.shape)
#             # gc1=self.bn_g(gc1)
#
#             gdc2 = F.relu(self.g_dconv2(gdc1))
#             print('gdc2.shape:', gdc2.shape)
#
#             zdc1 = F.relu(self.z_dconv1(z))
#             print('zdc1.shape', zdc1.shape)
#
#             m1 = torch.cat([gdc1, zdc1], 1)
#
#             print('m1.shape', m1.shape)
#             zdc2 = F.relu(self.z_dconv2(m1))
#
#             print('zdc2.shape', zdc2.shape)
#
#             # m2 = torch.cat([gdc2, zdc2], 1)
#             # print('m2.shape', m2.shape)
#
#             # m2_r = m2.view(-1, 48 * 38 * 38)
#             o = F.relu(self.z_dconv3(zdc2))
#             print(o.shape)
#             # print('output.shape', o.shape)
#             return o
#
# class ccGenerator(nn.Module):
#     def __init__(self):
#         super(ccGenerator,self).__init__()
#
#         self.use_gpu=torch.cuda.is_available()
#         self.g_conv1=nn.Conv2d(4,16,3,1,1)
#         self.z_conv1=nn.Conv2d(8,16,5,1,2)
#         self.z_pool1=nn.MaxPool2d(2)
#         self.z_conv2=nn.Conv2d(16,32,7,1,0)
#         self.z_conv3=nn.Conv2d(48,64,1,1,0)
#         self.z_conv4=nn.Conv2d(64,128,3,1,1)
#         self.fc1=nn.Linear(128*10*10,56*56)
#         # self.fc2=nn.Linear(6400,56*56)
#         self.lrelu=nn.LeakyReLU()
#
#         self.bn1=nn.BatchNorm2d(32)
#         self.bn2=nn.BatchNorm2d(64)
#         # self.bn3=nn.BatchNorm2d(128)
#     def forward(self, g,z):
#
#         if self.use_gpu:
#             g=g.view(-1,4,10,10).cuda()
#             gc1=self.g_conv1(g)
#             z=z.view(-1,8,32,32).cuda()
#             z1=self.lrelu(self.z_pool1(self.z_conv1(z)))
#             z2=self.lrelu(self.z_conv2(z1))
#             z2=self.bn1(z2)
#
#             _z3=torch.cat([gc1,z2],1)
#             z3=self.lrelu(self.z_conv3(_z3))
#             z3=self.bn2(z3)
#
#             z4=self.lrelu(self.z_conv4(z3))
#             z4=z4.view(-1,128*10*10).cuda()
#             o=self.fc1(z4)
#
#             # o = self.fc2(fc1)
#             return o.view(-1,1,56,56).cuda()
#         else:
#             g=g.view(-1,4,10,10)
#             gc1 = self.g_conv1(g)
#             z=z.view(-1,8,32,32)
#             z1=self.lrelu(self.z_pool1(self.z_conv1(z)))
#             z2=self.lrelu(self.z_conv2(z1))
#             print("z2.shape: ",z2.shape)
#             z2 = self.bn1(z2)
#             _z3=torch.cat([gc1,z2],1)
#             z3=self.lrelu(self.z_conv3(_z3))
#             z3 = self.bn2(z3)
#             z4=self.lrelu(self.z_conv4(z3))
#             print("z4.shape: ", z4.shape)
#             z4=z4.view(-1,128*10*10)
#             o=self.fc1(z4)
#             # print("fc1.shape: ", fc1.shape)
#             # o = self.fc2(fc1)
#             return o.view(-1,1,56,56)
