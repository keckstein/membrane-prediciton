import torch as t
import torch.nn as nn

class WeightMatrixWeightedBCE(nn.Module):
    def __init__(self, class_weights, weigh_with_matrix_sum =False):
        super(WeightMatrixWeightedBCE, self).__init__()

        self.class_weights = class_weights
        self.weigh_with_matrix_sum = weigh_with_matrix_sum


    def forward(self, y_pred, y_gt):

        cw = self.class_weights
        num_channels = y_pred.numpy().shape[1]

        assert len(cw) == num_channels, 'Class weight sets and number of channels have to match!'
        y_pred = t.clamp(y_pred, _epsilon, 1-_epsilon)

        w = y_gt[:,-1,:][:,None,:]
        loss = 0.
        if not self.weigh_with_matrix_sum:
            for c in range(num_channels):
                y_gt_c = y_gt[:,c,:][:,None,:]
                y_pred_c = y_pred[:,c,:][:,None,:]
                loss += w * -(cw[c][1] * y_gt_c * t.log(y_pred_c) + cw[c][0] * (1.0-y_gt_g) * t.log(-y_pred_c + 1.0))

        else:
            for c in range(num_channels):
                y_gt_c = y_gt[:,c,:][:,None,:]
                y_pred_c = y_pred[:,c,:][:,None,:]
                loss += t.sum(w) / w.nelement() * w * -(cw[c][1] * y_gt_c * t.log(y_pred_c) + cw[c][0] * (1.0 - y_gt_c) * t.log(- y_pred_c + 1.0))

        return t.mean(loss)


