from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

def filter_anomalies_by_local_support(pred, min_run_length=3):
    """
    Keep a predicted anomaly point only if there are at least min_run_length
    anomaly points inside a local window centered on that point.

    Example:
    min_run_length = 3  -> window size = 5
    so each point checks [i-2, i-1, i, i+1, i+2]
    and survives if sum(window) >= 3
    """
    if min_run_length <= 1:
        return pred.copy()

    pred = np.asarray(pred).astype(int)
    filtered = pred.copy()

    radius = min_run_length - 1   # for 3 -> radius 2, window size 5

    for i in range(len(pred)):
        if pred[i] == 1:
            left = max(0, i - radius)
            right = min(len(pred), i + radius + 1)
            local_sum = pred[left:right].sum()

            if local_sum < min_run_length:
                filtered[i] = 0

    return filtered

class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        pred = filter_anomalies_by_local_support(pred, min_run_length=self.args.min_run_length)

        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        # -------- RAW metrics --------
        raw_pred = pred.copy()
        raw_gt = gt.copy()

        raw_accuracy = accuracy_score(raw_gt, raw_pred)
        raw_precision, raw_recall, raw_f_score, _ = precision_recall_fscore_support(
            raw_gt, raw_pred, average='binary'
        )


        # -------- Adjusted metrics --------
        adj_gt, adj_pred = adjustment(gt.copy(), pred.copy())

        adj_pred = np.array(adj_pred)
        adj_gt = np.array(adj_gt)

        accuracy = accuracy_score(adj_gt, adj_pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            adj_gt, adj_pred, average='binary'
        )

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
            accuracy, precision, recall, f_score
        ))
        
        # =========================
        # CSV experiment logging
        # =========================
        log_dir = Path(__file__).resolve().parents[1] / "results"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "experiment_log.csv"

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "setting": setting,
            "task_name": self.args.task_name,
            "model_id": self.args.model_id,
            "model": self.args.model,
            "data": self.args.data,
            "root_path": self.args.root_path,

            # preprocessing / dataset metadata
            "turbine_id": getattr(self.args, "turbine_id", "T06"),
            "feature_mode": getattr(self.args, "feature_mode", "Avg"),
            "failure_window_hours": getattr(self.args, "failure_window_hours", 24),
            "imputation_method": getattr(self.args, "imputation_method", "time_linear_ffill_bfill"),

            # train/test setup
            "seq_len": self.args.seq_len,
            "label_len": self.args.label_len,
            "pred_len": self.args.pred_len,
            "features": self.args.features,
            "enc_in": self.args.enc_in,
            "dec_in": self.args.dec_in,
            "c_out": self.args.c_out,

            # model hyperparameters
            "d_model": self.args.d_model,
            "d_ff": self.args.d_ff,
            "n_heads": self.args.n_heads,
            "e_layers": self.args.e_layers,
            "d_layers": self.args.d_layers,
            "dropout": self.args.dropout,
            "moving_avg": self.args.moving_avg,

            # optimization
            "batch_size": self.args.batch_size,
            "train_epochs": self.args.train_epochs,
            "patience": self.args.patience,
            "learning_rate": self.args.learning_rate,
            "lradj": self.args.lradj,

            # anomaly config
            "anomaly_ratio": self.args.anomaly_ratio,
            "min_run_length": self.args.min_run_length,
            "threshold": float(threshold),

            # dataset sizes / outputs
            "train_windows": int(len(train_data)),
            "test_windows": int(len(test_data)),
            "raw_pred_points": int(len(raw_pred)),
            "raw_gt_points": int(len(raw_gt)),
            "adjusted_pred_points": int(len(adj_pred)),
            "adjusted_gt_points": int(len(adj_gt)),

            # anomaly counts
            "raw_pred_anomalies": int(raw_pred.sum()),
            "raw_gt_anomalies": int(raw_gt.sum()),
            "adjusted_pred_anomalies": int(adj_pred.sum()),
            "adjusted_gt_anomalies": int(adj_gt.sum()),

            # raw metrics
            "raw_accuracy": float(raw_accuracy),
            "raw_precision": float(raw_precision),
            "raw_recall": float(raw_recall),
            "raw_f1": float(raw_f_score),

            # adjusted metrics
            "adjusted_accuracy": float(accuracy),
            "adjusted_precision": float(precision),
            "adjusted_recall": float(recall),
            "adjusted_f1": float(f_score),
        }

        write_header = (not log_file.exists()) or (log_file.stat().st_size == 0)
        with open(log_file, "a", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(f"Logged experiment to: {log_file}")

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        return
