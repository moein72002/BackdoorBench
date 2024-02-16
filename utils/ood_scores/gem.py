# from __future__ import print_function
# import torch
# from torch.autograd import Variable
# import numpy as np
# from torch.autograd import Variable
#
# def get_gmm_score()
#
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False,
#                                           num_workers=args.prefetch, pin_memory=True)
#     num_batches = ood_num_examples // args.test_bs
#     #print("This is batch: ",num_batches)
#
#     temp_x = torch.rand(2,3,32,32)
#     temp_x = Variable(temp_x)
#     temp_x = temp_x.cuda() #PEYMAN
#     #temp_x=temp_x.cpu()
#     temp_list = net.feature_list(temp_x)[1]
#     num_output = len(temp_list)
#     feature_list = np.empty(num_output)
#     count = 0
#     for out in temp_list:
#         feature_list[count] = out.size(1)
#         count += 1
#
#     print('get sample mean and covariance', count)
#     sample_mean, precision = lib.sample_estimator(net, num_classes, feature_list, train_loader)
#     in_score,M_dist_1 = lib.get_GEM_Mahalanobis_score(net, test_loader, num_classes, sample_mean, precision, count-1, args.noise, num_batches, in_dist=True, GEM=1)
#     #my_frame=pd.DataFrame(M_dist_1)
#     #my_frame.to_csv("in_dist_11")
#     print(in_score[-3:], in_score[-103:-100])
#     ##PEYMAN-END
#
# def get_GEM_Mahalanobis_score(model, test_loader, num_classes, sample_mean, precision, layer_index, magnitude,
#                               num_batches, in_dist=False, GEM=0):
#     '''
#     If GEM!=1 then computes the proposed Mahalanobis confidence score on input dataset
#     If GEM=1 then computes the GEM score on input dataset
#     return: GEM or Mahalanobis score from layer_index
#     '''
#     model.eval()
#     Mahalanobis = []
#     M_dist = []
#
#     for batch_idx, (data, target) in enumerate(test_loader):
#         # print("this is target: ", target)
#         if batch_idx >= num_batches and in_dist is False:
#             break
#
#         data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, requires_grad=True), Variable(target)
#
#         out_features = model.intermediate_forward(data, layer_index)
#         out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
#         out_features = torch.mean(out_features, 2)
#
#         # compute Mahalanobis score
#         gaussian_score = 0
#         for i in range(num_classes):
#             batch_sample_mean = sample_mean[layer_index][i]
#             zero_f = out_features.data - batch_sample_mean
#             term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
#             if i == 0:
#                 gaussian_score = term_gau.view(-1, 1)
#             else:
#                 gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
#
#         # Input_processing
#         sample_pred = gaussian_score.max(1)[1]
#         batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
#         zero_f = out_features - Variable(batch_sample_mean)
#         pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
#         loss = torch.mean(-pure_gau)
#         loss.backward()
#
#         gradient = torch.ge(data.grad.data, 0)
#         gradient = (gradient.float() - 0.5) * 2
#         gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
#                              gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
#         gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
#                              gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
#         gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
#                              gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
#
#         # tempInputs = torch.add(data.data, -magnitude, gradient) #The useage is depreciated
#         tempInputs = torch.add(data.data, gradient, alpha=-magnitude)
#         with torch.no_grad():
#             noise_out_features = model.intermediate_forward(tempInputs, layer_index)
#         noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
#         noise_out_features = torch.mean(noise_out_features, 2)
#         noise_gaussian_score = 0
#         for i in range(num_classes):
#             batch_sample_mean = sample_mean[layer_index][i]
#             zero_f = noise_out_features.data - batch_sample_mean
#             term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
#             if i == 0:
#                 noise_gaussian_score = term_gau.view(-1, 1)
#             else:
#                 noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)
#         # print(noise_gaussian_score.data.cpu().numpy().shape)
#         my_target = target.data.cpu().numpy()
#         my_target = my_target.reshape(my_target.shape[0], -1)
#         my_score = noise_gaussian_score.data.cpu().numpy()
#         # print(my_target.shape,my_score.shape)
#         my_data = np.concatenate((my_score, my_target), axis=1)
#         M_dist.append(my_data)
#
#         # PEYMAN
#         if (GEM == 1):
#             noise_gaussian_score = torch.logsumexp(noise_gaussian_score, dim=1)
#         else:
#             noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
#         Mahalanobis.extend(-noise_gaussian_score.cpu().numpy())
#
#     return np.asarray(Mahalanobis, dtype=np.float32), np.concatenate(M_dist, axis=0)