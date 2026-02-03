import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from models_pytorch.utils import create_dataset, calculate_accuracy, calculate_ll, calculate_f1, ce_loss, DS_Combin


if torch.cuda.is_available():
    device = torch.device("cuda")  # 如果有可用的GPU，使用GPU
else:
    device = torch.device("cpu")  # 如果没有可用的GPU，使用CPU


def E_MNL_train(X_train, Q_train, y_train, model, N_EPOCHS=100, LR=0.001, BATCH_SIZE=50, drop=0.2, l2=0.01,
                VERBOSE=1, save_model=1, model_filename=str()):
    NUM_CHOICES = X_train.shape[2]
    NUM_CONT_VARS = X_train.shape[1]
    NUM_EMB_VARS = Q_train.shape[-1]
    UNIQUE_CATS = sorted(list(set(Q_train.reshape(-1))))
    NUM_UNIQUE_CATS = len(UNIQUE_CATS)

    e_mnl_model = model(NUM_CONT_VARS, NUM_EMB_VARS, NUM_CHOICES, NUM_UNIQUE_CATS, drop).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(e_mnl_model.parameters(), lr=LR, weight_decay=l2)

    if VERBOSE:
        print(e_mnl_model)

    train_dataset = create_dataset(X_train, Q_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    Loss, Acc, LL, F1 = [], [], [], []    ###########
    best_loss, best_acc, best_ll, best_epoch, best_f1 = float('inf'), 0, float('-inf'), 0, 0  ########
    best_model, best_model_weight = e_mnl_model, e_mnl_model.state_dict()
    for epoch in range(N_EPOCHS):
        total_loss, total_acc, total_ll, total_f1 = 0, 0, 0, 0  #############
        for i, data in enumerate(train_loader):
            X, Q, y = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            output = e_mnl_model(X, Q)
            # for name, param in e_mnl_model.named_parameters():
            #     if name == "embeddings.weight":
            #         l2_loss = torch.sum(param ** 2)
            loss = criterion(output, y)  # y.long()
            loss.backward()
            acc = calculate_accuracy(output, y)  # y.long()
            ll = calculate_ll(output, y)
            f1 = calculate_f1(output, y)  ##########
            clip_grad_norm_(e_mnl_model.parameters(), max_norm=50.0)  # 对梯度进行裁剪以防止过大的梯度导致内存溢出或者数值不稳定的问题
            optimizer.step()
            total_loss += loss.item()
            total_acc += acc
            total_ll += ll
            total_f1 += f1   ###########

        temp_loss, temp_acc, temp_ll = total_loss / len(train_loader), total_acc / len(train_loader), total_ll
        temp_f1 = total_f1 / len(train_loader)  #######
        Loss.append(temp_loss)
        Acc.append(temp_acc)
        LL.append(temp_ll)
        F1.append(temp_f1)  ######
        if epoch % 10 == 0 and VERBOSE:
            print('Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, LL: {:.0f}, f1: {:.4f}'.format(epoch, temp_loss, temp_acc, temp_ll, temp_f1))   ####

        if temp_acc > best_acc:
            best_loss, best_acc, best_ll, best_epoch, best_f1 = temp_loss, temp_acc, temp_ll, epoch, temp_f1 ####
            best_model, best_model_weight = e_mnl_model, e_mnl_model.state_dict()

    print("------------------------------------------------")
    print('best Epoch {}, best Loss: {:.4f}, best Accuracy: {:.4f}, best LL: {:.0f}, best f1: {:.4f}'.format(best_epoch, best_loss, best_acc, best_ll, best_f1))  ###
    print("------------------------------------------------")

    if save_model:
        torch.save(best_model_weight, str(best_epoch)+str("_")+model_filename)
        print('Model saved as {}'.format(str(best_epoch)+str("_")+model_filename))

    return best_model, best_loss, best_acc, best_ll, best_f1, Loss  ###


def MNL_train(
    X_train,
    Q_train,
    y_train,
    model,
    N_EPOCHS=100,
    LR=0.001,
    BATCH_SIZE=50,
    l2=0.01,
    VERBOSE=1,
    save_model=1,
    model_filename=str(),
):
    NUM_CHOICES = X_train.shape[2]
    NUM_CONT_VARS = X_train.shape[1]

    mnl_model = model(NUM_CONT_VARS, NUM_CHOICES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mnl_model.parameters(), lr=LR, weight_decay=l2)

    if VERBOSE:
        print(mnl_model)

    train_dataset = create_dataset(X_train, Q_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    Loss, Acc, LL, F1 = [], [], [], []
    best_loss, best_acc, best_ll, best_epoch, best_f1 = float("inf"), 0, float("-inf"), 0, 0
    best_model, best_model_weight = mnl_model, mnl_model.state_dict()
    for epoch in range(N_EPOCHS):
        total_loss, total_acc, total_ll, total_f1 = 0, 0, 0, 0
        for i, data in enumerate(train_loader):
            X, _, y = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            output = mnl_model(X)
            loss = criterion(output, y)
            loss.backward()
            acc = calculate_accuracy(output, y)
            ll = calculate_ll(output, y)
            f1 = calculate_f1(output, y)
            clip_grad_norm_(mnl_model.parameters(), max_norm=50.0)
            optimizer.step()
            total_loss += loss.item()
            total_acc += acc
            total_ll += ll
            total_f1 += f1

        temp_loss = total_loss / len(train_loader)
        temp_acc = total_acc / len(train_loader)
        temp_ll = total_ll
        temp_f1 = total_f1 / len(train_loader)
        Loss.append(temp_loss)
        Acc.append(temp_acc)
        LL.append(temp_ll)
        F1.append(temp_f1)
        if epoch % 10 == 0 and VERBOSE:
            print(
                "Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, LL: {:.0f}, f1: {:.4f}".format(
                    epoch, temp_loss, temp_acc, temp_ll, temp_f1
                )
            )

        if temp_acc > best_acc:
            best_loss, best_acc, best_ll, best_epoch, best_f1 = temp_loss, temp_acc, temp_ll, epoch, temp_f1
            best_model, best_model_weight = mnl_model, mnl_model.state_dict()

    print("------------------------------------------------")
    print(
        "best Epoch {}, best Loss: {:.4f}, best Accuracy: {:.4f}, best LL: {:.0f}, best f1: {:.4f}".format(
            best_epoch, best_loss, best_acc, best_ll, best_f1
        )
    )
    print("------------------------------------------------")

    if save_model:
        torch.save(best_model_weight, str(best_epoch) + str("_") + model_filename)
        print("Model saved as {}".format(str(best_epoch) + str("_") + model_filename))

    return best_model, best_loss, best_acc, best_ll, best_f1, Loss


def EL_MNL_train(X_train, Q_train, y_train, model, extra_emb_dims, n_nodes, N_EPOCHS=100, LR=0.001, l2=0.01,
                 BATCH_SIZE=50, drop=0.2, VERBOSE=1, save_model=1, model_filename=str()):
    NUM_CHOICES = X_train.shape[2]
    NUM_CONT_VARS = X_train.shape[1]
    NUM_EMB_VARS = Q_train.shape[-1]
    UNIQUE_CATS = sorted(list(set(Q_train.reshape(-1))))
    NUM_UNIQUE_CATS = len(UNIQUE_CATS)

    el_mnl_model = model(NUM_CONT_VARS, NUM_EMB_VARS, NUM_CHOICES, NUM_UNIQUE_CATS, extra_emb_dims, n_nodes, drop).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(el_mnl_model.parameters(), lr=LR, weight_decay=l2)

    if VERBOSE:
        print(el_mnl_model)

    train_dataset = create_dataset(X_train, Q_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    Loss, Acc, LL, F1 = [], [], [], []
    best_loss, best_acc, best_ll, best_epoch, best_f1 = float('inf'), 0, float('-inf'), 0, 0
    best_model, best_model_weight = el_mnl_model, el_mnl_model.state_dict()
    for epoch in range(N_EPOCHS):
        total_loss, total_acc, total_ll, total_f1 = 0, 0, 0, 0
        for i, data in enumerate(train_loader):
            X, Q, y = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            output = el_mnl_model(X, Q)
            loss = criterion(output, y)  # y.long()
            acc = calculate_accuracy(output, y)  # y.long()
            ll = calculate_ll(output, y)
            f1 = calculate_f1(output, y)
            loss.backward()
            clip_grad_norm_(el_mnl_model.parameters(), max_norm=50.0)  # 对梯度进行裁剪以防止过大的梯度导致内存溢出或者数值不稳定的问题
            optimizer.step()
            total_loss += loss.item()
            total_acc += acc
            total_ll += ll
            total_f1 += f1

        temp_loss, temp_acc, temp_ll = total_loss / len(train_loader), total_acc / len(train_loader), total_ll
        temp_f1 = total_f1 / len(train_loader)
        Loss.append(temp_loss)
        Acc.append(temp_acc)
        LL.append(temp_ll)
        F1.append(temp_f1)
        if epoch % 10 == 0 and VERBOSE:
            print('Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, LL: {:.0f}, f1: {:.4f}'.format(epoch, temp_loss, temp_acc, temp_ll, temp_f1))

        if temp_acc > best_acc:
            best_loss, best_acc, best_ll, best_epoch, best_f1 = temp_loss, temp_acc, temp_ll, epoch, temp_f1
            best_model, best_model_weight = el_mnl_model, el_mnl_model.state_dict()

    print("------------------------------------------------")
    print('best Epoch {}, best Loss: {:.4f}, best Accuracy: {:.4f}, best LL: {:.0f}, best f1: {:.4f}'.format(best_epoch, best_loss, best_acc, best_ll, best_f1))
    print("------------------------------------------------")

    if save_model:
        torch.save(best_model_weight, str(best_epoch)+str("_")+model_filename)
        print('Model saved as {}'.format(str(best_epoch)+str("_")+model_filename))

    return best_model, best_loss, best_acc, best_ll, best_f1, Loss


def L_MNL_train(X_train, Q_train, y_train, model, n_nodes, N_EPOCHS=100, LR=0.001, l2=0.01,
                BATCH_SIZE=50, VERBOSE=1, save_model=1, model_filename=str()):
    NUM_CHOICES = X_train.shape[2]
    NUM_CONT_VARS = X_train.shape[1]
    NUM_EMB_VARS = Q_train.shape[-1]

    l_mnl_model = model(NUM_CONT_VARS, NUM_EMB_VARS, NUM_CHOICES, n_nodes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(l_mnl_model.parameters(), lr=LR, weight_decay=l2)

    if VERBOSE:
        print(l_mnl_model)

    train_dataset = create_dataset(X_train, Q_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    Loss, Acc, LL, F1 = [], [], [], []
    best_loss, best_acc, best_ll, best_epoch, best_f1 = float('inf'), 0, float('-inf'), 0, 0
    best_model, best_model_weight = l_mnl_model, l_mnl_model.state_dict()
    for epoch in range(N_EPOCHS):
        total_loss, total_acc, total_ll, total_f1 = 0, 0, 0, 0
        for i, data in enumerate(train_loader):
            X, Q, y = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            output = l_mnl_model(X, Q)
            loss = criterion(output, y)  # y.long()
            acc = calculate_accuracy(output, y)  # y.long()
            ll = calculate_ll(output, y)
            f1 = calculate_f1(output, y)
            loss.backward()
            clip_grad_norm_(l_mnl_model.parameters(), max_norm=50.0)  # 对梯度进行裁剪以防止过大的梯度导致内存溢出或者数值不稳定的问题
            optimizer.step()
            total_loss += loss.item()
            total_acc += acc
            total_ll += ll
            total_f1 += f1

        temp_loss, temp_acc, temp_ll = total_loss / len(train_loader), total_acc / len(train_loader), total_ll
        temp_f1 = total_f1 / len(train_loader)
        Loss.append(temp_loss)
        Acc.append(temp_acc)
        LL.append(temp_ll)
        F1.append(temp_f1)
        if epoch % 10 == 0 and VERBOSE:
            print('Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, LL: {:.0f}, f1: {:.4f}'.format(epoch, temp_loss, temp_acc, temp_ll, temp_f1))

        if temp_acc > best_acc:
            best_loss, best_acc, best_ll, best_epoch, best_f1 = temp_loss, temp_acc, temp_ll, epoch, temp_f1
            best_model, best_model_weight = l_mnl_model, l_mnl_model.state_dict()

    print("------------------------------------------------")
    print('best Epoch {}, best Loss: {:.4f}, best Accuracy: {:.4f}, best LL: {:.0f}, best f1: {:.4f}'.format(best_epoch, best_loss, best_acc, best_ll, best_f1))
    print("------------------------------------------------")

    if save_model:
        torch.save(best_model_weight, str(best_epoch)+str("_")+model_filename)
        print('Model saved as {}'.format(str(best_epoch)+str("_")+model_filename))

    return best_model, best_loss, best_acc, best_ll, best_f1, Loss


def TE_MNL_train(X_train, Q_train, y_train, model, lambda_epochs, N_EPOCHS=100, LR=0.001, BATCH_SIZE=50, drop=0.2, l2=0.01,
                 VERBOSE=1, save_model=1, model_filename=str()):
    NUM_CHOICES = X_train.shape[2]
    NUM_CONT_VARS = X_train.shape[1]
    NUM_EMB_VARS = Q_train.shape[-1]
    UNIQUE_CATS = sorted(list(set(Q_train.reshape(-1))))
    NUM_UNIQUE_CATS = len(UNIQUE_CATS)
    lambda_epochs = lambda_epochs

    te_mnl_model = model(NUM_CONT_VARS, NUM_EMB_VARS, NUM_CHOICES, NUM_UNIQUE_CATS, lambda_epochs, drop).to(device)
    optimizer = optim.Adam(te_mnl_model.parameters(), lr=LR, weight_decay=l2)

    if VERBOSE:
        print(te_mnl_model)

    train_dataset = create_dataset(X_train, Q_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    Loss, Acc, F1 = [], [], []
    best_loss, best_acc, best_epoch, best_f1 = float('inf'), 0, 0, 0
    best_model, best_model_weight = te_mnl_model, te_mnl_model.state_dict()
    for epoch in range(N_EPOCHS):
        total_loss, total_acc, total_f1 = 0, 0, 0
        for i, data in enumerate(train_loader):
            X, Q, y = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            evidence = te_mnl_model(X, Q)
            loss = 0
            alpha = dict()
            for v_num in range(len(evidence)):
                alpha[v_num] = evidence[v_num] + 1
                loss += ce_loss(y, alpha[v_num], NUM_CHOICES, epoch, lambda_epochs)
            alpha_a = DS_Combin(alpha, NUM_CHOICES)
            evidence_a = alpha_a - 1
            loss += ce_loss(y, alpha_a, NUM_CHOICES, epoch, lambda_epochs)
            loss = torch.mean(loss)

            loss.backward()
            acc = calculate_accuracy(evidence_a, y)
            f1 = calculate_f1(evidence_a, y)
            clip_grad_norm_(te_mnl_model.parameters(), max_norm=50.0)  # 对梯度进行裁剪以防止过大的梯度导致内存溢出或者数值不稳定的问题
            optimizer.step()
            total_loss += loss
            total_acc += acc
            total_f1 += f1

        temp_loss, temp_acc, temp_f1 = total_loss / len(train_loader), total_acc / len(train_loader), total_f1 / len(train_loader)
        Loss.append(temp_loss)
        Acc.append(temp_acc)
        F1.append(temp_f1)
        if epoch % 10 == 0 and VERBOSE:
            print('Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, f1: {:.4f}'.format(epoch, temp_loss, temp_acc, temp_f1))

        if temp_acc > best_acc:
            best_loss, best_acc, best_epoch, best_f1 = temp_loss, temp_acc, epoch, temp_f1
            best_model, best_model_weight = te_mnl_model, te_mnl_model.state_dict()

    print("------------------------------------------------")
    print('best Epoch {}, best Loss: {:.4f}, best Accuracy: {:.4f}, best f1: {:.4f}'.format(best_epoch, best_loss, best_acc, best_f1))
    print("------------------------------------------------")

    if save_model:
        torch.save(best_model_weight, str(best_epoch)+str("_")+model_filename)
        print('Model saved as {}'.format(str(best_epoch)+str("_")+model_filename))

    return best_model, best_loss, best_acc, best_f1, Loss


def TEL_MNL_train(
    X_train,
    Q_train,
    y_train,
    model,
    lambda_epochs,
    extra_emb_dims,
    n_nodes,
    N_EPOCHS=100,
    LR=0.001,
    BATCH_SIZE=50,
    drop=0.2,
    l2=0.01,
    VERBOSE=1,
    save_model=1,
    model_filename=str(),
    use_emb_extra=True,
):
    NUM_CHOICES = X_train.shape[2]
    NUM_CONT_VARS = X_train.shape[1]
    NUM_EMB_VARS = Q_train.shape[-1]
    UNIQUE_CATS = sorted(list(set(Q_train.reshape(-1))))
    NUM_UNIQUE_CATS = len(UNIQUE_CATS)
    lambda_epochs = lambda_epochs

    tel_mnl_model = model(
        NUM_CONT_VARS,
        NUM_EMB_VARS,
        NUM_CHOICES,
        NUM_UNIQUE_CATS,
        extra_emb_dims,
        n_nodes,
        lambda_epochs,
        drop,
        use_emb_extra=use_emb_extra,
    ).to(device)
    optimizer = optim.Adam(tel_mnl_model.parameters(), lr=LR, weight_decay=l2)

    if VERBOSE:
        print(tel_mnl_model)

    train_dataset = create_dataset(X_train, Q_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    Loss, Acc, F1 = [], [], []
    best_loss, best_acc, best_epoch, best_f1 = float('inf'), 0, 0, 0
    best_model, best_model_weight = tel_mnl_model, tel_mnl_model.state_dict()
    for epoch in range(N_EPOCHS):
        total_loss, total_acc, total_f1 = 0, 0, 0
        for i, data in enumerate(train_loader):
            X, Q, y = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            evidence = tel_mnl_model(X, Q)
            loss = 0
            alpha = dict()
            for v_num in range(len(evidence)):
                alpha[v_num] = evidence[v_num] + 1
                loss += ce_loss(y, alpha[v_num], NUM_CHOICES, epoch, lambda_epochs)
            alpha_a = DS_Combin(alpha, NUM_CHOICES)
            evidence_a = alpha_a - 1
            loss += ce_loss(y, alpha_a, NUM_CHOICES, epoch, lambda_epochs)
            loss = torch.mean(loss)

            loss.backward()
            acc = calculate_accuracy(evidence_a, y)
            f1 = calculate_f1(evidence_a, y)
            clip_grad_norm_(tel_mnl_model.parameters(), max_norm=50.0)  # 对梯度进行裁剪以防止过大的梯度导致内存溢出或者数值不稳定的问题
            optimizer.step()
            total_loss += loss
            total_acc += acc
            total_f1 += f1

        temp_loss, temp_acc, temp_f1 = total_loss / len(train_loader), total_acc / len(train_loader), total_f1 / len(train_loader)
        Loss.append(temp_loss)
        Acc.append(temp_acc)
        F1.append(temp_f1)
        if epoch % 10 == 0 and VERBOSE:
            print('Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, f1: {:.4f}'.format(epoch, temp_loss, temp_acc, temp_f1))

        if temp_acc > best_acc:
            best_loss, best_acc, best_epoch, best_f1 = temp_loss, temp_acc, epoch, temp_f1
            best_model, best_model_weight = tel_mnl_model, tel_mnl_model.state_dict()

    print("------------------------------------------------")
    print('best Epoch {}, best Loss: {:.4f}, best Accuracy: {:.4f}, best f1: {:.4f}'.format(best_epoch, best_loss, best_acc, best_f1))
    print("------------------------------------------------")

    if save_model:
        torch.save(best_model_weight, str(best_epoch)+str("_")+model_filename)
        print('Model saved as {}'.format(str(best_epoch)+str("_")+model_filename))

    return best_model, best_loss, best_acc, best_f1, Loss
