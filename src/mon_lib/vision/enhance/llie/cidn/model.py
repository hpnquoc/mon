import torch
import torch.nn as nn

import networks


class DRIT(nn.Module):
  
  def __init__(self, opts):
    super(DRIT, self).__init__()

    # parameters
    lr = 0.0001
    lr_dcontent = lr / 2.5
    self.nz = 8
    self.concat = opts.concat
    self.no_ms = opts.no_ms
    self.content_share = opts.content_share
    self.attri_share = opts.attribute_share
    self.vgg = opts.vgg
    self.gpu = opts.gpu

    # perceptual
    if opts.vgg > 0:
      self.vgg_loss = networks.PerceptualLoss(opts)
      # if self.opt.IN_vgg:
      #   self.vgg_patch_loss = networks.PerceptualLoss(opts)
      #   self.vgg_patch_loss.cuda()
      self.vgg_loss.cuda()
      self.vgg_net = networks.load_vgg16("../model", self.gpu)
      self.vgg_net.eval()
      for param in self.vgg_net.parameters():
        param.requires_grad = False

    # discriminators
    if opts.dis_scale > 1:
      self.disA = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disA2 = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB2 = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
    else:
      self.disA = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disA2 = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB2 = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
    self.disContent = networks.DiscriminatorContent()

    # encoders
    if self.content_share:
      self.enc_c = networks.E_content_share(opts.input_dim_a, opts.input_dim_b)
    else:
      self.enc_c = networks.E_content(opts.input_dim_a, opts.input_dim_b)

    if self.concat:
      if self.attri_share:
        self.enc_a = networks.E_attr_concat_share(opts.input_dim_a, opts.input_dim_b, self.nz, \
                                            norm_layer=None, nl_layer=networks.get_non_linearity(layer_type='lrelu'))
      else:
        self.enc_a = networks.E_attr_concat(opts.input_dim_a, opts.input_dim_b, self.nz, \
                                            norm_layer=None, nl_layer=networks.get_non_linearity(layer_type='lrelu'))
    else:
      self.enc_a = networks.E_attr(opts.input_dim_a, opts.input_dim_b, self.nz)

    # generator
    if self.concat:
      self.gen = networks.G_concat(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
    else:
      self.gen = networks.G(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

    # optimizers
    self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disA2_opt = torch.optim.Adam(self.disA2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB2_opt = torch.optim.Adam(self.disB2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

    # Setup the loss function for training
    self.criterionL1 = torch.nn.L1Loss()

  def initialize(self):
    self.disA.apply(networks.gaussian_weights_init)
    self.disB.apply(networks.gaussian_weights_init)
    self.disA2.apply(networks.gaussian_weights_init)
    self.disB2.apply(networks.gaussian_weights_init)
    self.disContent.apply(networks.gaussian_weights_init)
    self.gen.apply(networks.gaussian_weights_init)
    self.enc_c.apply(networks.gaussian_weights_init)
    self.enc_a.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.disA_sch       = networks.get_scheduler(self.disA_opt      , opts, last_ep)
    self.disB_sch       = networks.get_scheduler(self.disB_opt      , opts, last_ep)
    self.disA2_sch      = networks.get_scheduler(self.disA2_opt     , opts, last_ep)
    self.disB2_sch      = networks.get_scheduler(self.disB2_opt     , opts, last_ep)
    self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)
    self.enc_c_sch      = networks.get_scheduler(self.enc_c_opt     , opts, last_ep)
    self.enc_a_sch      = networks.get_scheduler(self.enc_a_opt     , opts, last_ep)
    self.gen_sch        = networks.get_scheduler(self.gen_opt       , opts, last_ep)

  def setgpu(self, gpu):
    self.gpu = gpu
    self.disA.cuda(self.gpu)
    self.disB.cuda(self.gpu)
    self.disA2.cuda(self.gpu)
    self.disB2.cuda(self.gpu)
    self.disContent.cuda(self.gpu)
    self.enc_c.cuda(self.gpu)
    self.enc_a.cuda(self.gpu)
    self.gen.cuda(self.gpu)

  def get_z_random(self, batchSize, nz, random_type='gauss'):
    z = torch.randn(batchSize, nz).cuda(self.gpu)
    return z

  def test_forward(self, image, a2b=True):
    self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
    if a2b:
        self.z_content = self.enc_c.forward_a(image)
        output = self.gen.forward_b(self.z_content, self.z_random)
    else:
        self.z_content = self.enc_c.forward_b(image)
        output = self.gen.forward_a(self.z_content, self.z_random)
    return output

  def test_forward_transfer(self, image_a, image_b, a2b=True):
    self.z_content_a, self.z_content_b = self.enc_c.forward(image_a, image_b)
    if self.concat:
      self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(image_a, image_b)
      std_a = self.logvar_a.mul(0.5).exp_()
      eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_a = eps.mul(std_a).add_(self.mu_a)
      std_b = self.logvar_b.mul(0.5).exp_()
      eps = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_b = eps.mul(std_b).add_(self.mu_b)
    else:
      self.z_attr_a, self.z_attr_b = self.enc_a.forward(image_a, image_b)
    if a2b:
      output = self.gen.forward_b(self.z_content_a, self.z_attr_b)
    else:
      output = self.gen.forward_a(self.z_content_b, self.z_attr_a)
    return output

  def forward(self):
    # input images
    half_size = 1
    real_A = self.input_A
    real_B = self.input_B
    self.real_A_encoded = real_A[0:half_size]
    self.real_A_random = real_A[half_size:]
    self.real_B_encoded = real_B[0:half_size]
    self.real_B_random = real_B[half_size:]

    # get encoded z_c
    self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)
    # fea_a = self.real_A_encoded.mean(1, keepdim=True)
    # fea_a = fea_a.cpu().detach().numpy()
    # fea_a = np.squeeze(fea_a, axis=0)
    # fea_a = np.transpose(fea_a, (1, 2, 0))
    # # plt.imshow(fea_a, cmap=plt.get_cmap('gray'))
    # # io.show()
    # fea_b = self.real_B_encoded.mean(1, keepdim=True)
    # fea_b = fea_b.cpu().detach().numpy()
    # fea_b = np.squeeze(fea_b, axis=0)
    # fea_b = np.transpose(fea_b, (1, 2, 0))
    # # plt.imshow(fea_b, cmap=plt.get_cmap('gray'))
    # # io.show()
    # fea_a = self.z_content_a.mean(1, keepdim=True)
    # fea_a = fea_a.cpu().detach().numpy()
    # fea_a = np.squeeze(fea_a, axis=0)
    # fea_a = np.transpose(fea_a, (1, 2, 0))
    # # plt.imshow(fea_a, cmap=plt.get_cmap('gray'))
    # io.show()
    # fea_b = self.z_content_b.mean(1, keepdim=True)
    # fea_b = fea_b.cpu().detach().numpy()
    # fea_b = np.squeeze(fea_b, axis=0)
    # fea_b = np.transpose(fea_b, (1, 2, 0))
    # plt.imshow(fea_b, cmap=plt.get_cmap('gray'))
    # io.show()
    # # plt.imshow(c_b, cmap=plt.get_cmap('gray'))
    # #     # io.show()
    # get encoded z_a
    if self.concat:
      self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)
      std_a = self.logvar_a.mul(0.5).exp_()
      eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_a = eps_a.mul(std_a).add_(self.mu_a)
      std_b = self.logvar_b.mul(0.5).exp_()
      eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_b = eps_b.mul(std_b).add_(self.mu_b)
    else:
      self.z_attr_a, self.z_attr_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)

    # get random z_a
    self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')
    if not self.no_ms:
      self.z_random2 = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')

    if not self.no_ms:
      input_content_forB = torch.cat((self.z_content_b, self.z_content_b, self.z_content_b), 0)
      input_content_forA = torch.cat((self.z_content_a, self.z_content_a, self.z_content_a), 0)
      input_attr_forA = torch.cat((self.z_attr_a, self.z_random, self.z_random2), 0)
      input_attr_forB = torch.cat((self.z_attr_b, self.z_random, self.z_random2), 0)
      output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
      output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
      self.fake_AA_encoded, self.fake_AA_random, self.fake_AA_random2 = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
      self.fake_BB_encoded, self.fake_BB_random, self.fake_BB_random2 = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)
    else:
      input_content_forB = torch.cat((self.z_content_b, self.z_content_b), 0)
      input_content_forA = torch.cat((self.z_content_a,  self.z_content_a), 0)
      input_attr_forA = torch.cat((self.z_attr_a, self.z_random),0)
      input_attr_forB = torch.cat((self.z_attr_b, self.z_random),0)
      output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
      output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
      self.fake_AA_encoded, self.fake_AA_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
      self.fake_BB_encoded, self.fake_BB_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

    # first cross translation
    if not self.no_ms:
      input_content_forA = torch.cat((self.z_content_b, self.z_content_b, self.z_content_b),0)
      input_content_forB = torch.cat((self.z_content_a, self.z_content_a, self.z_content_a),0)
      input_attr_forA = torch.cat((self.z_attr_a, self.z_random, self.z_random2),0)
      input_attr_forB = torch.cat((self.z_attr_b, self.z_random, self.z_random2),0)
      output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
      output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
      self.fake_A_encoded, self.fake_A_random, self.fake_A_random2 = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
      self.fake_B_encoded, self.fake_B_random, self.fake_B_random2 = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)
    else:
      input_content_forA = torch.cat((self.z_content_b, self.z_content_b),0)
      input_content_forB = torch.cat((self.z_content_a, self.z_content_a),0)
      input_attr_forA = torch.cat((self.z_attr_a, self.z_random),0)
      input_attr_forB = torch.cat((self.z_attr_b, self.z_random),0)
      output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
      output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
      self.fake_A_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
      self.fake_B_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

    # # get reconstructed encoded z_c
    # self.z_content_recon_b, self.z_content_recon_a = self.enc_c.forward(self.fake_A_encoded, self.fake_B_encoded)

    # get reconstructed encoded z_a
    # if self.concat:
    #   self.mu_recon_a, self.logvar_recon_a, self.mu_recon_b, self.logvar_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)
    #   std_a = self.logvar_recon_a.mul(0.5).exp_()
    #   eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
    #   self.z_attr_recon_a = eps_a.mul(std_a).add_(self.mu_recon_a)
    #   std_b = self.logvar_recon_b.mul(0.5).exp_()
    #   eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
    #   self.z_attr_recon_b = eps_b.mul(std_b).add_(self.mu_recon_b)
    # else:
    #   self.z_attr_recon_a, self.z_attr_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)
    #
    # # second cross translation
    # self.fake_A_recon = self.gen.forward_a(self.z_content_recon_a, self.z_attr_recon_a)
    # self.fake_B_recon = self.gen.forward_b(self.z_content_recon_b, self.z_attr_recon_b)

    # for display
    self.image_display = torch.cat((self.real_A_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(), \
                                    self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(), \
                                    self.real_B_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(), \
                                    self.fake_A_random[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu()), dim=0)

    # # for latent regression
    # if self.concat:
    #   self.mu2_a, _, self.mu2_b, _ = self.enc_a.forward(self.fake_A_random, self.fake_B_random)
    # else:
    #   self.z_attr_random_a, self.z_attr_random_b = self.enc_a.forward(self.fake_A_random, self.fake_B_random)

  def forward_content(self):
    half_size = 1
    self.real_A_encoded = self.input_A[0:half_size]
    self.real_B_encoded = self.input_B[0:half_size]
    # get encoded z_c
    self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)

  def update_D_content(self, image_a, image_b):
    self.input_A = image_a
    self.input_B = image_b
    self.forward_content()
    self.disContent_opt.zero_grad()
    loss_D_Content       = self.backward_contentD(self.z_content_a, self.z_content_b)
    self.disContent_loss = loss_D_Content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def update_D(self, image_a, image_b):
    self.input_A = image_a
    self.input_B = image_b
    self.forward()

    # update disA
    self.disA_opt.zero_grad()
    loss_D1_A = self.backward_D(self.disA, self.real_A_encoded, self.fake_A_encoded)
    self.disA_loss = loss_D1_A.item()
    self.disA_opt.step()

    # update disA2
    self.disA2_opt.zero_grad()
    loss_D2_A = self.backward_D(self.disA2, self.real_A_random, self.fake_A_random)
    self.disA2_loss = loss_D2_A.item()
    if not self.no_ms:
      loss_D2_A2 = self.backward_D(self.disA2, self.real_A_random, self.fake_A_random2)
      self.disA2_loss += loss_D2_A2.item()
    self.disA2_opt.step()

    # update disB
    self.disB_opt.zero_grad()
    loss_D1_B = self.backward_D(self.disB, self.real_B_encoded, self.fake_B_encoded)
    self.disB_loss = loss_D1_B.item()
    self.disB_opt.step()

    # update disB2
    self.disB2_opt.zero_grad()
    loss_D2_B = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random)
    self.disB2_loss = loss_D2_B.item()
    if not self.no_ms:
      loss_D2_B2 = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random2)
      self.disB2_loss += loss_D2_B2.item()
    self.disB2_opt.step()

    # update disContent
    self.disContent_opt.zero_grad()
    loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
    self.disContent_loss = loss_D_Content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def backward_D(self, netD, real, fake):
    pred_fake = netD.forward(fake.detach())
    pred_real = netD.forward(real)
    loss_D = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all0 = torch.zeros_like(out_fake).cuda(self.gpu)
      all1 = torch.ones_like(out_real).cuda(self.gpu)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      loss_D += ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def backward_contentD(self, imageA, imageB):
    pred_fake = self.disContent.forward(imageA.detach())
    pred_real = self.disContent.forward(imageB.detach())
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all1 = torch.ones((out_real.size(0))).cuda(self.gpu)
      all0 = torch.zeros((out_fake.size(0))).cuda(self.gpu)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
    loss_D = ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def update_EG(self):
    # update G, Ec, Ea
    self.enc_c_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_EG()
    self.enc_c_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()

    # update G, Ec
    self.enc_c_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_G_alone()
    self.enc_c_opt.step()
    self.gen_opt.step()

  def backward_EG(self):
    # content Ladv for generator
    loss_G_GAN_Acontent = self.backward_G_GAN_content(self.z_content_a)
    loss_G_GAN_Bcontent = self.backward_G_GAN_content(self.z_content_b)

    # Ladv for generator
    loss_G_GAN_A = self.backward_G_GAN(self.fake_A_encoded, self.disA)
    loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.disB)

    # KL loss - z_a
    if self.concat:
      kl_element_a = self.mu_a.pow(2).add_(self.logvar_a.exp()).mul_(-1).add_(1).add_(self.logvar_a)
      loss_kl_za_a = torch.sum(kl_element_a).mul_(-0.5) * 0.01
      kl_element_b = self.mu_b.pow(2).add_(self.logvar_b.exp()).mul_(-1).add_(1).add_(self.logvar_b)
      loss_kl_za_b = torch.sum(kl_element_b).mul_(-0.5) * 0.01
    else:
      loss_kl_za_a = self._l2_regularize(self.z_attr_a) * 0.01
      loss_kl_za_b = self._l2_regularize(self.z_attr_b) * 0.01

    # KL loss - z_c
    loss_kl_zc_a = self._l2_regularize(self.z_content_a) * 0.01
    loss_kl_zc_b = self._l2_regularize(self.z_content_b) * 0.01

    # cross cycle consistency loss
    # loss_G_L1_A = self.criterionL1(self.fake_A_recon, self.real_A_encoded) * 10
    # loss_G_L1_B = self.criterionL1(self.fake_B_recon, self.real_B_encoded) * 10
    loss_content_ab = self.criterionL1(self.z_content_a, self.z_content_b) * 10
    loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded) * 10
    loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded) * 10

    # perceptual loss
    if self.vgg > 0:
      loss_vgg_a = self.vgg_loss.compute_vgg_loss(self.vgg_net, self.fake_A_encoded, self.input_A)
      loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg_net, self.fake_B_encoded, self.input_B)
      loss_vgg_random_a = self.vgg_loss.compute_vgg_loss(self.vgg_net, self.fake_A_random, self.input_A)
      loss_vgg_random_b = self.vgg_loss.compute_vgg_loss(self.vgg_net, self.fake_B_random, self.input_B)
      loss_G = loss_G_GAN_A + loss_G_GAN_B + \
               loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
               loss_vgg_random_a + loss_vgg_random_b + \
               loss_vgg_a + loss_vgg_b + \
               loss_G_L1_AA + loss_G_L1_BB + loss_content_ab +\
               loss_kl_zc_a + loss_kl_zc_b + \
               loss_kl_za_a + loss_kl_za_b

      loss_G.backward(retain_graph=True)

      self.loss_vgg_a = loss_vgg_a.item()
      self.loss_vgg_b = loss_vgg_b.item()
    else:
      loss_G = loss_G_GAN_A + loss_G_GAN_B + \
               loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
               loss_G_L1_AA + loss_G_L1_BB + loss_content_ab +\
               loss_kl_zc_a + loss_kl_zc_b + \
               loss_kl_za_a + loss_kl_za_b

      loss_G.backward(retain_graph=True)

    self.content_loss_ab = loss_content_ab.item()
    self.gan_loss_a = loss_G_GAN_A.item()
    self.gan_loss_b = loss_G_GAN_B.item()
    self.gan_loss_acontent = loss_G_GAN_Acontent.item()
    self.gan_loss_bcontent = loss_G_GAN_Bcontent.item()
    self.kl_loss_za_a = loss_kl_za_a.item()
    self.kl_loss_za_b = loss_kl_za_b.item()
    self.kl_loss_zc_a = loss_kl_zc_a.item()
    self.kl_loss_zc_b = loss_kl_zc_b.item()
    self.l1_recon_AA_loss = loss_G_L1_AA.item()
    self.l1_recon_BB_loss = loss_G_L1_BB.item()
    self.G_loss = loss_G.item()

  def backward_G_GAN_content(self, data):
    outs = self.disContent.forward(data)
    for out in outs:
      outputs_fake = nn.functional.sigmoid(out)
      all_half = 0.5*torch.ones((outputs_fake.size(0))).cuda(self.gpu)
      ad_loss = nn.functional.binary_cross_entropy(outputs_fake, all_half)
    return ad_loss

  def backward_G_GAN(self, fake, netD=None):
    outs_fake = netD.forward(fake)
    loss_G = 0
    for out_a in outs_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
    return loss_G

  def backward_G_alone(self):
    # Ladv for generator
    loss_G_GAN2_A = self.backward_G_GAN(self.fake_A_random, self.disA2)
    loss_G_GAN2_B = self.backward_G_GAN(self.fake_B_random, self.disB2)
    if not self.no_ms:
      loss_G_GAN2_A2 = self.backward_G_GAN(self.fake_A_random2, self.disA2)
      loss_G_GAN2_B2 = self.backward_G_GAN(self.fake_B_random2, self.disB2)

    # mode seeking loss for A-->B and B-->A
    if not self.no_ms:
      lz_AB = torch.mean(torch.abs(self.fake_B_random2 - self.fake_B_random)) / torch.mean(torch.abs(self.z_random2 - self.z_random))
      lz_BA = torch.mean(torch.abs(self.fake_A_random2 - self.fake_A_random)) / torch.mean(torch.abs(self.z_random2 - self.z_random))
      eps = 1 * 1e-5
      loss_lz_AB = 1 / (lz_AB + eps)
      loss_lz_BA = 1 / (lz_BA + eps)
    # # latent regression loss
    # if self.concat:
    #   loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - self.z_random)) * 10
    #   loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 10
    # else:
    #   loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - self.z_random)) * 10
    #   loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10

    loss_z_L1 = loss_G_GAN2_A + loss_G_GAN2_B
    if not self.no_ms:
      loss_z_L1 += (loss_G_GAN2_A2 + loss_G_GAN2_B2)
      loss_z_L1 += (loss_lz_AB + loss_lz_BA)

    # perceptual loss
    # if self.vgg > 0:
    #   loss_vgg_a = self.vgg_loss.compute_vgg_loss(self.vgg_net, self.fake_A_encoded, self.input_A)
    #   loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg_net, self.fake_B_encoded, self.input_B)
    #   loss_z_L1 += loss_vgg_a + loss_vgg_b

    loss_z_L1.backward()
    if not self.no_ms:
      self.gan2_loss_a = loss_G_GAN2_A.item() + loss_G_GAN2_A2.item()
      self.gan2_loss_b = loss_G_GAN2_B.item() + loss_G_GAN2_B2.item()
      self.lz_AB = loss_lz_AB.item()
      self.lz_BA = loss_lz_BA.item()
    else:
      self.gan2_loss_a = loss_G_GAN2_A.item()
      self.gan2_loss_b = loss_G_GAN2_B.item()
    # if self.vgg > 0:
    #   self.loss_vgg_a = loss_vgg_a.item()
    #   self.loss_vgg_b = loss_vgg_b.item()

  def update_lr(self):
    self.disA_sch.step()
    self.disB_sch.step()
    self.disA2_sch.step()
    self.disB2_sch.step()
    self.disContent_sch.step()
    self.enc_c_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()

  def _l2_regularize(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir)
    # weight
    if train:
      self.disA.load_state_dict(checkpoint['disA'])
      self.disA2.load_state_dict(checkpoint['disA2'])
      self.disB.load_state_dict(checkpoint['disB'])
      self.disB2.load_state_dict(checkpoint['disB2'])
      self.disContent.load_state_dict(checkpoint['disContent'])
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    self.enc_a.load_state_dict(checkpoint['enc_a'])
    self.gen.load_state_dict(checkpoint['gen'])
    # optimizer
    if train:
      self.disA_opt.load_state_dict(checkpoint['disA_opt'])
      self.disA2_opt.load_state_dict(checkpoint['disA2_opt'])
      self.disB_opt.load_state_dict(checkpoint['disB_opt'])
      self.disB2_opt.load_state_dict(checkpoint['disB2_opt'])
      self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
      self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
      self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
    return checkpoint['ep'], checkpoint['total_it']

  def save(self, filename, ep, total_it):
    state = {
             'disA': self.disA.state_dict(),
             'disA2': self.disA2.state_dict(),
             'disB': self.disB.state_dict(),
             'disB2': self.disB2.state_dict(),
             'disContent': self.disContent.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'enc_a': self.enc_a.state_dict(),
             'gen': self.gen.state_dict(),
             'disA_opt': self.disA_opt.state_dict(),
             'disA2_opt': self.disA2_opt.state_dict(),
             'disB_opt': self.disB_opt.state_dict(),
             'disB2_opt': self.disB2_opt.state_dict(),
             'disContent_opt': self.disContent_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    torch.save(state, filename)
    return

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_A_encoded).detach()
    images_b = self.normalize_image(self.real_B_encoded).detach()
    images_a1 = self.normalize_image(self.fake_A_encoded).detach()
    # images_a2 = self.normalize_image(self.fake_A_random).detach()
    # images_a3 = self.normalize_image(self.fake_A_recon).detach()
    images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
    images_b1 = self.normalize_image(self.fake_B_encoded).detach()
    # images_b2 = self.normalize_image(self.fake_B_random).detach()
    # images_b3 = self.normalize_image(self.fake_B_recon).detach()
    images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
    row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_a4[0:1, ::]),3)
    row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_b4[0:1, ::]),3)
    return torch.cat((row1,row2),2)

  def normalize_image(self, x):
    return x[:,0:3,:,:]
