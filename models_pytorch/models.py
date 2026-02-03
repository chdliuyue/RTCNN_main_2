import torch
import torch.nn as nn
import torch.nn.functional as F


# Keras input (batch_size, height, width, channels)
# pytorch input (batch_size, channels, height, width)
def MNL(vars_num, choices_num, logits_activation='softmax'):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=(vars_num, 1), stride=1, padding=0, bias=False)
            self.activation = F.softmax if logits_activation == 'softmax' else F.relu

        def forward(self, x):
            x = self.conv(x)
            x = x.reshape(-1, choices_num)
            x = self.activation(x, dim=1)
            return x

    model = Model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(model)

    return model

# (5, 12, 3, 81)
def E_MNL(cont_vars_num, emb_vars_num, choices_num, unique_cats_num, drop=0.2,
          pos_constraint=True, logits_activation='softmax'):
    """ E-MNL: Multinomial Logit model as a CNN
               with interpretable embeddings """
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.emb_size = choices_num
            self.pos_constraint = pos_constraint
            self.embeddings = nn.Embedding(unique_cats_num, self.emb_size, max_norm=1, norm_type=2.0)
            self.dropout = nn.Dropout(drop)

            if self.pos_constraint:
                self.utilities1 = nn.Conv2d(1, 1, kernel_size=(emb_vars_num, 1),
                                            stride=1, padding=0, bias=False, dtype=torch.float32)
                self.utilities2 = nn.Conv2d(1, 1, kernel_size=(cont_vars_num, 1),
                                            stride=1, padding=0, bias=False, dtype=torch.float32)
            else:
                self.utilities = nn.Conv2d(1, 1, kernel_size=(emb_vars_num + cont_vars_num, 1),
                                           stride=1, padding=0, bias=False, dtype=torch.float32)

            self.activation = F.softmax if logits_activation == 'softmax' else F.relu

        def forward(self, main_input, emb_input):
            emb = self.embeddings(emb_input)
            emb = self.dropout(emb)
            emb = emb.reshape(-1, 1, emb_vars_num, choices_num)
            main = main_input.permute(0, 3, 1, 2)
            emb = emb.float()
            main = main.float()

            if self.pos_constraint:
                utilities1 = self.utilities1(emb)
                self.utilities1.weight.data.clamp_(min=0)
                utilities2 = self.utilities2(main)
                utilities = torch.add(utilities1, utilities2)
            else:
                utilities = torch.cat((emb, main), dim=2)
                utilities = self.utilities(utilities)

            output = utilities.reshape(-1, choices_num)
            output = self.activation(output, dim=1)
            return output

    model = Model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(model)

    return model

# (5, 12, 3, 81, 2, 15)
def EL_MNL(cont_vars_num, emb_vars_num, choices_num, unique_cats_num,
           extra_emb_dims, n_nodes, drop=0.2, pos_constraint=True, logits_activation='softmax'):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.emb_size = choices_num + extra_emb_dims
            self.pos_constraint = pos_constraint
            self.embeddings = nn.Embedding(unique_cats_num, self.emb_size, max_norm=1, norm_type=2.0)
            self.dropout = nn.Dropout(drop)

            self.dense = nn.Conv2d(1, n_nodes, kernel_size=(emb_vars_num*extra_emb_dims, 1), stride=1, padding=0, dtype=torch.float32)
            self.relu3 = nn.ReLU()
            self.fc3 = nn.Linear(n_nodes, choices_num, dtype=torch.float32)

            if self.pos_constraint:
                self.utilities1 = nn.Conv2d(1, 1, kernel_size=(emb_vars_num, 1),
                                            stride=1, padding=0, bias=False, dtype=torch.float32)
                self.utilities2 = nn.Conv2d(1, 1, kernel_size=(cont_vars_num, 1),
                                            stride=1, padding=0, bias=False, dtype=torch.float32)
            else:
                self.utilities = nn.Conv2d(1, 1, kernel_size=(emb_vars_num + cont_vars_num, 1),
                                           stride=1, padding=0, bias=False, dtype=torch.float32)

            self.activation = F.softmax if logits_activation == 'softmax' else F.relu

        def forward(self, main_input, emb_input):
            emb = self.embeddings(emb_input)
            emb = self.dropout(emb)
            emb = emb.float()

            emb_extra = emb[:, :, choices_num:]
            emb_extra = emb_extra.reshape(-1, 1, emb_vars_num*extra_emb_dims, 1)
            emb_extra = self.dense(emb_extra)
            emb_extra = self.relu3(emb_extra)
            emb_extra = emb_extra.reshape(-1, n_nodes)
            emb_extra = self.fc3(emb_extra)
            emb_extra = emb_extra.reshape(-1, 1, 1, choices_num)

            emb = emb[:, :, :choices_num]
            emb = emb.reshape(-1, 1, emb_vars_num, choices_num)

            main = main_input.permute(0, 3, 1, 2)
            main = main.float()
            if self.pos_constraint:
                utilities1 = self.utilities1(emb)
                self.utilities1.weight.data.clamp_(min=0)
                utilities2 = self.utilities2(main)
                # 对应上面网络的解释
                # 输出是(n, 1, 1, 3) cat操作会输出(n, 1, 2, 3)
                utilities = torch.add(utilities1, utilities2)
            else:
                utilities = torch.cat((emb, main), dim=2)
                utilities = self.utilities(utilities)
            output = torch.add(utilities, emb_extra)
            output = output.reshape(-1, choices_num)
            output = self.activation(output, dim=1)
            return output

    model = Model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(model)

    return model

# (5, 12, 3, 15)
def L_MNL(cont_vars_num, emb_vars_num, choices_num, n_nodes):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.dense1 = nn.Linear(emb_vars_num, n_nodes, dtype=torch.float32)
            self.bn1 = nn.BatchNorm1d(n_nodes)
            self.dense2 = nn.Linear(n_nodes, choices_num, dtype=torch.float32)

            self.utilities2 = nn.Conv2d(1, 1, kernel_size=(cont_vars_num, 1),
                                        stride=1, padding=0, bias=False, dtype=torch.float32)

            self.activation = F.softmax

        def forward(self, main_input, emb_input):
            emb = emb_input.float()
            emb = self.dense1(emb)
            emb = self.bn1(emb)
            emb = self.dense2(emb)

            main = main_input.permute(0, 3, 1, 2)
            main = main.float()
            utilities2 = self.utilities2(main)
            utilities2 = utilities2.reshape(-1, choices_num)

            output = torch.add(emb, utilities2)
            output = self.activation(output, dim=1)
            return output

    model = Model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(model)

    return model

def TE_MNL(cont_vars_num, emb_vars_num, choices_num, unique_cats_num, lambda_epochs=1, drop=0.2):

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.emb_size = choices_num
            self.lambda_epochs = lambda_epochs
            self.embeddings = nn.Embedding(unique_cats_num, self.emb_size, max_norm=1, norm_type=2.0)
            self.dropout = nn.Dropout(drop)

            self.fc1 = nn.Linear(emb_vars_num*choices_num, emb_vars_num*choices_num)
            # self.bn1 = nn.BatchNorm1d(emb_vars_num * choices_num)
            self.fc2 = nn.Linear(cont_vars_num*choices_num, cont_vars_num*choices_num)

            self.utilities1 = nn.Conv2d(1, 1, kernel_size=(emb_vars_num, 1),
                                        stride=1, padding=0, bias=False, dtype=torch.float32)
            self.utilities2 = nn.Conv2d(1, 1, kernel_size=(cont_vars_num, 1),
                                        stride=1, padding=0, bias=False, dtype=torch.float32)

            self.activation1 = nn.Softplus()
            self.activation2 = nn.Softplus()

        def forward(self, main_input, emb_input):
            emb = self.embeddings(emb_input)
            emb = self.dropout(emb)
            emb = emb.float()
            emb = emb.reshape(-1, emb_vars_num * choices_num)
            emb = self.fc1(emb)
            # emb = self.bn1(emb)
            emb = emb.reshape(-1, 1, emb_vars_num, choices_num)

            main = main_input.float()
            main = main.reshape(-1, cont_vars_num*choices_num)
            main = self.fc2(main)
            main = main.reshape(-1, 1, cont_vars_num, choices_num)

            utilities1 = self.utilities1(emb)
            utilities2 = self.utilities2(main)
            self.utilities1.weight.data.clamp_(min=0)
            output1 = utilities1.reshape(-1, choices_num)
            output2 = utilities2.reshape(-1, choices_num)
            evidence = dict()
            evidence[0] = self.activation1(output1)
            evidence[1] = self.activation2(output2)

            return evidence

    model = Model()

    return model


def TEL_MNL(cont_vars_num, emb_vars_num, choices_num, unique_cats_num,
            extra_emb_dims, n_nodes, lambda_epochs=1, drop=0.2, use_emb_extra=True):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.emb_size = choices_num + extra_emb_dims
            self.lambda_epochs = lambda_epochs
            self.use_emb_extra = use_emb_extra
            self.embeddings = nn.Embedding(unique_cats_num, self.emb_size, max_norm=1, norm_type=2.0)
            self.dropout = nn.Dropout(drop)

            if self.use_emb_extra:
                self.dense = nn.Conv2d(
                    1,
                    n_nodes,
                    kernel_size=(emb_vars_num * extra_emb_dims, 1),
                    stride=1,
                    padding=0,
                    dtype=torch.float32,
                )
                self.relu3 = nn.ReLU()
                self.fc3 = nn.Linear(n_nodes, choices_num, dtype=torch.float32)
            else:
                self.dense = None
                self.relu3 = None
                self.fc3 = None

            self.fc1 = nn.Linear(emb_vars_num * choices_num, emb_vars_num * choices_num)
            self.fc2 = nn.Linear(cont_vars_num * choices_num, cont_vars_num * choices_num)
            # self.bn1 = nn.BatchNorm1d(emb_vars_num * choices_num)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()

            self.utilities1 = nn.Conv2d(1, 1, kernel_size=(emb_vars_num, 1),
                                        stride=1, padding=0, bias=False, dtype=torch.float32)
            self.utilities2 = nn.Conv2d(1, 1, kernel_size=(cont_vars_num, 1),
                                        stride=1, padding=0, bias=False, dtype=torch.float32)

            self.activation1 = nn.Softplus()
            self.activation2 = nn.Softplus()
            self.activation3 = nn.Softplus()

        def forward(self, main_input, emb_input):
            emb = self.embeddings(emb_input)
            emb = self.dropout(emb)
            emb = emb.float()

            emb_extra = None
            if self.use_emb_extra:
                emb_extra = emb[:, :, choices_num:]
                emb_extra = emb_extra.reshape(-1, 1, emb_vars_num * extra_emb_dims, 1)
                emb_extra = self.dense(emb_extra)
                emb_extra = self.relu3(emb_extra)
                emb_extra = emb_extra.reshape(-1, n_nodes)
                emb_extra = self.fc3(emb_extra)

            emb = emb[:, :, :choices_num]
            emb = emb.reshape(-1, emb_vars_num * choices_num)
            emb = self.fc1(emb)
            # emb = self.bn1(emb)
            emb = self.relu1(emb)
            emb = emb.reshape(-1, 1, emb_vars_num, choices_num)

            main = main_input.float()
            main = main.reshape(-1, cont_vars_num * choices_num)
            main = self.fc2(main)
            main = self.relu2(main)
            main = main.reshape(-1, 1, cont_vars_num, choices_num)

            utilities1 = self.utilities1(emb)
            utilities2 = self.utilities2(main)
            self.utilities1.weight.data.clamp_(min=0.0)
            output1 = utilities1.reshape(-1, choices_num)
            output2 = utilities2.reshape(-1, choices_num)
            evidence = dict()
            evidence[0] = self.activation1(output1)
            evidence[1] = self.activation2(output2)
            if self.use_emb_extra:
                evidence[2] = self.activation3(emb_extra)

            return evidence

    model = Model()

    return model

if __name__ == "__main__":
    MNL(17, 3)
    E_MNL(5, 12, 3, 81)
    EL_MNL(5, 12, 3, 81, 2, 15)
    L_MNL(5, 12, 3, 15)
