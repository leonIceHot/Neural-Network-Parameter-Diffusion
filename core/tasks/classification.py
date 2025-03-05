import pdb
import hydra.utils
import torch
import torch.nn as nn
import datetime
import glob
import omegaconf
import json

from sklearn.metrics import roc_auc_score
from .base_task import BaseTask
from torch.utils.tensorboard import SummaryWriter
from core.utils import *
from core.data.vision_dataset import VisionData
from core.data.parameters import PData
from core.utils.utils import *

# For Chestmnist
class CFTask(BaseTask):
    def __init__(self, config, **kwargs):
        super(CFTask, self).__init__(config, **kwargs)
        self.train_loader = self.task_data.train_dataloader()
        self.eval_loader = self.task_data.val_dataloader()
        self.test_loader = self.task_data.test_dataloader()
        self.dataset = "chestmnist"
        self.writer = SummaryWriter('runs/' + self.dataset)

    def init_task_data(self):
        return VisionData(self.cfg.data)

    # override the abstract method in base_task.py
    def set_param_data(self):
        param_data = PData(self.cfg.param)
        self.model = param_data.get_model()
        self.train_layer = param_data.get_train_layer()
        return param_data

    def test_g_model(self, input):
        net = self.model
        train_layer = self.train_layer
        param = torch.squeeze(input)

        target_num = sum(torch.numel(module) for name, module in net.named_parameters() if name in train_layer)
        params_num = param.shape[0]

        # 參數一致性檢查
        # assert (target_num == params_num), f"Parameter count mismatch: {target_num} != {params_num}"

        # 還原模型
        model = partial_reverse_tomodel(param, net, train_layer).to(param.device)
        model.eval()

        test_loss = 0.0
        correct = 0
        total = 0
        output_list = []

        with torch.no_grad():
            for num, (data, target) in enumerate(self.test_loader, start=1):
                #print(f"Num: {num}")
                data, target = data.cuda(), target.cuda()

                # 確保 target 是浮點型
                target = target.float()

                # 模型前向傳播
                output = model(data)

                # Binary Cross-Entropy Loss
                loss_fn = nn.BCEWithLogitsLoss()
                test_loss += loss_fn(output, target).item() * data.size(0)

                # 將輸出轉換為二元分類標籤 (0 或 1)
                predicted = (output.sigmoid() > 0.5).float()

                # 計算正確數量
                batch_correct = (predicted.long() == target.long()).sum().item()
                #print(f"Batch correct: {batch_correct}")
                correct += batch_correct

                # 確認 total 是元素總數
                total += target.numel()

                # 儲存預測結果
                output_list.extend(predicted.cpu().numpy().tolist())

        # 計算平均損失
        test_loss /= total

        # 計算準確度
        acc = 100. * correct / total if total > 0 else 0.0

        print(f"Correct: {correct}, Total: {total}, Accuracy: {acc:.2f}%")

        del model
        return acc, test_loss, output_list

    def val_g_model(self, input):
        net = self.model
        train_layer = self.train_layer
        param = torch.squeeze(input)

        # 計算需要更新的參數數量
        target_num = sum(torch.numel(module) for name, module in net.named_parameters() if name in train_layer)
        params_num = param.shape[0]

        # 驗證參數數量是否一致
        assert target_num == params_num, f"Parameter count mismatch: {target_num} != {params_num}"

        # 還原模型參數
        model = partial_reverse_tomodel(param, net, train_layer).to(param.device)
        model.eval()

        test_loss = 0.0
        correct = 0
        total = 0
        output_list = []

        with torch.no_grad():
            for num, (data, target) in enumerate(self.train_loader, start=1):
                #print(f"Num: {num}")
                data, target = data.cuda(), target.cuda()

                # 確保 target 和 output 的數據類型一致
                target = target.float()

                # 模型前向傳播
                output = model(data)

                # 計算 Binary Cross-Entropy Loss
                loss_fn = nn.BCEWithLogitsLoss()
                test_loss += loss_fn(output, target).item() * data.size(0)

                # 將輸出轉換為二元分類標籤 (0 或 1)
                predicted = (output.sigmoid() > 0.5).float()

                # 儲存預測結果
                output_list.extend(predicted.cpu().numpy().tolist())

                # 計算正確預測數量
                correct += (predicted == target).sum().item()

                # 計算總元素數量，而不是 batch 大小
                total += target.numel()

        # 計算平均損失
        test_loss /= total

        # 計算準確率
        acc = 100. * correct / total if total > 0 else 0.0

        #print(f"Total: {total}")
        #print(f"Correct: {correct}")
        #print(f"Accuracy: {acc:.2f}%")
        print(f"Correct: {correct}, Total: {total}, Accuracy: {acc:.2f}%")

        return acc, test_loss, output_list

    # override the abstract method in base_task.py, you obtain the model data for generation
    def train_for_data(self):
        net = self.build_model()
        optimizer = self.build_optimizer(net)
        criterion = nn.CrossEntropyLoss()
        scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer)
        epoch = self.cfg.epoch
        save_num = self.cfg.save_num_model
        all_epoch = epoch + save_num

        best_acc = 0
        train_loader = self.train_loader
        eval_loader = self.eval_loader
        train_layer = self.cfg.train_layer

        if train_layer == 'all':
            train_layer = [name for name, module in net.named_parameters()]

        data_path = getattr(self.cfg, 'save_root', 'param_data')

        tmp_path = os.path.join(data_path, 'tmp_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        # tmp_path = os.path.join(data_path, 'tmp')
        final_path = os.path.join(data_path, self.cfg.data.dataset)

        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(final_path, exist_ok=True)

        save_model_accs = []
        parameters = []

        net = net.cuda()
        for i in range(0, all_epoch):
            self.train(net, criterion, optimizer, train_loader, i)
            acc = self.test(net, criterion, eval_loader, i)
            #print(f"acc:{acc}, best acc:{best_acc}")
            best_acc = max(acc, best_acc)

            if i == (epoch - 1):
                print("saving the model")
                torch.save(net, os.path.join(tmp_path, "whole_model.pth"))
                fix_partial_model(train_layer, net)
                parameters = []

            if i >= epoch:
                parameters.append(state_part(train_layer, net))
                save_model_accs.append(acc)
                if len(parameters) == 10 or i == all_epoch - 1:
                    torch.save(parameters, os.path.join(tmp_path, "p_data_{}.pt".format(i)))
                    parameters = []

            scheduler.step()
        print("training over")

        pdata = []
        for file in glob.glob(os.path.join(tmp_path, "p_data_*.pt")):
            buffers = torch.load(file)
            for buffer in buffers:
                param = []
                for key in buffer.keys():
                    if key in train_layer:
                        param.append(buffer[key].data.reshape(-1))
                param = torch.cat(param, 0)
                pdata.append(param)
        batch = torch.stack(pdata)
        mean = torch.mean(batch, dim=0)
        std = torch.std(batch, dim=0)

        # check the memory of p_data
        useage_gb = get_storage_usage(tmp_path)
        print(f"path {tmp_path} storage usage: {useage_gb:.2f} GB")

        state_dic = {
            'pdata': batch.cpu().detach(),
            'mean': mean.cpu(),
            'std': std.cpu(),
            'model': torch.load(os.path.join(tmp_path, "whole_model.pth")),
            'train_layer': train_layer,
            'performance': save_model_accs,
            'cfg': config_to_dict(self.cfg)
        }

        torch.save(state_dic, os.path.join(final_path, "data.pt"))
        json_state = {
            'cfg': config_to_dict(self.cfg),
            'performance': save_model_accs

        }
        json.dump(json_state, open(os.path.join(final_path, "config.json"), 'w'))

        # copy the code file(the file) in state_save_dir
        shutil.copy(os.path.abspath(__file__), os.path.join(final_path,
                                                            os.path.basename(__file__)))

        # delete the tmp_path
        shutil.rmtree(tmp_path)
        print("data process over")
        return {'save_path': final_path}

    def train(self, net, criterion, optimizer, trainloader, epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct_predictions = 0
        total_predictions = 0

        all_outputs = []
        all_targets = []

        # Tensorboard summary
        dataset_sizes = len(trainloader)

        running_loss = 0.0
        running_corrects = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.float().cuda()
            #targets = target.squeeze()
            #print("target", target.shape)
            #print("targets",targets.shape)
            #self.writer.add_scalar('y_true', targets, epoch)
            # [8,1,28,28]
            #print(f"Image shape: {inputs.shape}")
            # [8,14]
            #print(f"Target shape: {targets.shape}")

            optimizer.zero_grad()
            outputs = net(inputs)

            #all_outputs.append(outputs.cpu())
            #all_targets.append(targets.cpu())

            # [8,14]
            #print(f"outputs shape: {outputs.shape}")

            #loss = criterion(outputs, targets)
            loss = nn.BCEWithLogitsLoss()(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            #_, predicted = outputs.max(1)
            #self.writer.add_scalar('y_label', predicted, epoch)

            # Calculate acc
            #total += targets.size(0)
            predicted = (outputs.sigmoid() > 0.5).float().cuda()
            #print(f"Predicted:{predicted.device}")
            #print(f"Target:{targets.device}")
            correct_predictions += predicted.eq(targets).float().sum().item()
            total_predictions += targets.numel()
            #total_predictions += targets.size()[0]

            # Calculate auc
            # all_outputs = torch.cat(all_outputs).detach().numpy()
            # all_targets = torch.cat(all_targets).detach().numpy()
            # auc = roc_auc_score(all_targets, all_outputs,average='macro')

            #epoch_loss     = '%.3f'%(running_loss / (batch_idx + 1))
            #epoch_acc  = 100. * correct / total
            epoch_loss = '%.3f' % (running_loss / len(trainloader))
            epoch_acc = correct_predictions / total_predictions

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (running_loss / (batch_idx + 1),
                            epoch_acc,
                            correct_predictions,
                            total_predictions))

            # Record training loss and accuracy for each phase
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
            #self.writer.add_scalar('Train/AUC', auc, epoch)
            self.writer.flush()

    def test(self, net, criterion, testloader, epoch):
        global best_acc
        net.eval()
        test_loss = 0
        #correct = 0
        #total = 0
        correct_predictions = 0
        total_predictions = 0

        # Tensorboard summary
        dataset_size = len(testloader)

        running_loss = 0.0
        running_corrects = 0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.float().cuda()
                #targets = targets.squeeze()
                #print(f"Image shape: {inputs.shape}")
                #print(f"Target shape: {targets.shape}")

                # (batch_size, 14)
                outputs = net(inputs)
                #outputs = net(inputs).sigmoid()
                #all_targets.append(targets.cpu())
                #all_outputs.append(outputs.cpu())
                #print(f"Output shape: {outputs.shape}")

                #loss = criterion(outputs, targets)
                loss = nn.BCEWithLogitsLoss()(outputs, targets)

                running_loss += loss.item()
                #_, predicted = outputs.max(1)
                #total += targets.size(0)
                #correct += predicted.eq(targets).sum().item()

                # Calculate acc
                predicted = (outputs.sigmoid() > 0.5).float().cuda()
                correct_predictions += predicted.eq(targets).float().sum().item()
                total_predictions += targets.numel()
                #total_predictions += targets.size(0)
                #total_predictions += targets.size()[0]

                # Calculate auc
                #all_outputs = torch.cat(all_outputs).detach().numpy()
                #all_targets = torch.cat(all_targets).detach().numpy()
                #auc         = roc_auc_score(all_targets, all_outputs, average='macro')

                #epoch_loss = '%.3f'%(running_loss / (batch_idx + 1))
                #epoch_loss = '%.3f' % (running_loss / len(testloader))
                #epoch_acc = 100. * correct / total

                epoch_loss = '%.3f' % (running_loss / len(testloader))
                epoch_acc = correct_predictions / total_predictions

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (running_loss / (batch_idx + 1), epoch_acc, correct_predictions, total_predictions))

                # Tensorboard summary
                # Record training loss and accuracy for each phase
                self.writer.add_scalar('Val/Loss', epoch_loss, epoch)
                self.writer.add_scalar('Val/Accuracy', epoch_acc, epoch)
                #self.writer.add_scalar('Val/AUC', auc, epoch)
                self.writer.flush()

            return epoch_acc



