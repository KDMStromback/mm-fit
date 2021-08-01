import os
import tqdm
import copy
import pprint
import torch
import torch.nn as nn
import pandas as pd
from utils import utils
import torchvision.transforms as transforms
from utils.data_transforms import Unit, Resample
from utils.dataset import MMFit, SequentialStridedSampler
from torch.utils.data import RandomSampler, ConcatDataset
from model.conv_ae import ConvAutoencoder
from model.multimodal_ae import MultimodalAutoencoder
from model.multimodal_ar import MultimodalFcClassifier


################
# Configuration
################

args = utils.parse_args()
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(vars(args))
torch.backends.cudnn.benchmark = True

ACTIONS = ['squats', 'lunges', 'bicep_curls', 'situps', 'pushups', 'tricep_extensions', 'dumbbell_rows',
           'jumping_jacks', 'dumbbell_shoulder_press', 'lateral_shoulder_raises', 'non_activity']
TRAIN_W_IDs = ['01', '02', '03', '04', '06', '07', '08', '16', '17', '18']
VAL_W_IDs = ['14', '15', '19']
if args.unseen_test_set:
    TEST_W_IDs = ['00', '05', '12', '13', '20']
else:
    TEST_W_IDs = ['09', '10', '11']
# All modalities available in MM-Fit
MODALITIES = ['sw_l_acc', 'sw_l_gyr', 'sw_l_hr', 'sw_r_acc', 'sw_r_gyr', 'sw_r_hr', 'sp_l_acc', 'sp_l_gyr',
              'sp_l_mag', 'sp_r_acc', 'sp_r_gyr', 'sp_r_mag', 'eb_l_acc', 'eb_l_gyr', 'pose_2d', 'pose_3d']
# We use a subset of all modalities in this demo.
MODALITIES_SUBSET = ['sw_l_acc', 'sw_l_gyr', 'sw_r_acc', 'sw_r_gyr', 'sp_r_acc', 'sp_r_gyr', 'eb_l_acc', 'eb_l_gyr',
                     'pose_3d']

exp_name = args.name
output_path = args.output
if not os.path.exists(output_path):
    os.makedirs(output_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

window_stride = int(args.window_stride * args.skeleton_sampling_rate)
skeleton_window_length = int(args.window_length * args.skeleton_sampling_rate)
sensor_window_length = int(args.window_length * args.target_sensor_sampling_rate)

# Set model training hyperparameters
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

################
# Create data loaders
################

data_transforms = {
        'skeleton': transforms.Compose([
            Unit()
        ]),
        'sensor': transforms.Compose([
            Resample(target_length=sensor_window_length)
        ])
    }

train_datasets, val_datasets, test_datasets = [], [], []
for w_id in TRAIN_W_IDs + VAL_W_IDs + TEST_W_IDs:
    modality_filepaths = {}
    workout_path = os.path.join(args.data, 'w' + w_id)
    files = os.listdir(workout_path)
    label_path = None
    for file in files:
        if 'labels' in file:
            label_path = os.path.join(workout_path, file)
            continue
        for modality_type in MODALITIES_SUBSET:
            if modality_type in file:
                modality_filepaths[modality_type] = os.path.join(workout_path, file)
    if label_path is None:
        raise Exception('Error: Label file not found for workout {}.'.format(w_id))

    if w_id in TRAIN_W_IDs:
        train_datasets.append(MMFit(modality_filepaths, label_path, args.window_length, skeleton_window_length,
                                    sensor_window_length, skeleton_transform=data_transforms['skeleton'],
                                    sensor_transform=data_transforms['sensor']))
    elif w_id in VAL_W_IDs:
        val_datasets.append(MMFit(modality_filepaths, label_path, args.window_length, skeleton_window_length,
                                  sensor_window_length, skeleton_transform=data_transforms['skeleton'],
                                  sensor_transform=data_transforms['sensor']))
    elif w_id in TEST_W_IDs:
        test_datasets.append(MMFit(modality_filepaths, label_path, args.window_length, skeleton_window_length,
                                   sensor_window_length, skeleton_transform=data_transforms['skeleton'],
                                   sensor_transform=data_transforms['sensor']))
    else:
        raise Exception('Error: Workout {} not assigned to train, test, or val datasets'.format(w_id))

train_dataset = ConcatDataset(train_datasets)
val_dataset = ConcatDataset(val_datasets)
test_dataset = ConcatDataset(test_datasets)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           sampler=RandomSampler(train_dataset), pin_memory=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                         sampler=SequentialStridedSampler(val_dataset, window_stride),
                                         pin_memory=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                          sampler=SequentialStridedSampler(test_dataset, window_stride),
                                          pin_memory=True, num_workers=4)

################
# Instantiate model
################

sw_l_acc_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[3, 3, 1],
                                 kernel_size=11, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

sw_l_gyr_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[1, 1, 1],
                                 kernel_size=3, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

sw_r_acc_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[3, 3, 1],
                                 kernel_size=11, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

sw_r_gyr_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[1, 1, 1],
                                 kernel_size=3, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

eb_acc_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[3, 3, 1],
                               kernel_size=11, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

eb_gyr_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[1, 1, 1],
                               kernel_size=3, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

sp_acc_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[1, 1, 1],
                               kernel_size=11, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

sp_gyr_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[1, 1, 1],
                               kernel_size=3, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

skel_model = ConvAutoencoder(input_size=(skeleton_window_length, 16), input_ch=3, dim=2, layers=3, grouped=[3, 3, 1],
                             kernel_size=11, kernel_stride=(2, 1), return_embeddings=True).to(device, non_blocking=True)

multimodal_ae_f_in = 4800
multimodal_ae_model = MultimodalAutoencoder(f_in=multimodal_ae_f_in, sw_l_acc=sw_l_acc_model, sw_l_gyr=sw_l_gyr_model,
                                            sw_r_acc=sw_r_acc_model, sw_r_gyr=sw_r_gyr_model, eb_acc=eb_acc_model,
                                            eb_gyr=eb_gyr_model, sp_acc=sp_acc_model, sp_gyr=sp_gyr_model,
                                            skel=skel_model, layers=args.ae_layers, hidden_units=args.ae_hidden_units,
                                            f_embedding=args.embedding_units, dropout=args.ae_dropout,
                                            return_embeddings=True).to(device, non_blocking=True)

if args.multimodal_ae_wp != "":
    multi_ae_params = torch.load(args.multimodal_ae_wp, map_location=device)
    multimodal_ae_model.load_state_dict(multi_ae_params['model_state_dict'])

f_in = args.embedding_units
model = MultimodalFcClassifier(f_in=f_in, num_classes=args.num_classes, multimodal_ae_model=multimodal_ae_model,
                               layers=args.layers, hidden_units=args.hidden_units,
                               dropout=args.dropout).to(device, non_blocking=True)

if args.model_wp != "":
    model_params = torch.load(args.model_wp, map_location=device)
    model.load_state_dict(model_params['model_state_dict'])

################
# Training
################

if args.multimodal_ae_wp != "":
    fc_params = []
    pre_trained_params = []
    for child in list(model.children()):
        if isinstance(child, MultimodalAutoencoder):
            for grandchild in child.children():
                if isinstance(grandchild, ConvAutoencoder):
                    for c in grandchild.children():
                        pre_trained_params.extend(list(c.parameters()))
                else:
                    pre_trained_params.extend(list(grandchild.parameters()))
        else:
            fc_params.extend(list(child.parameters()))

    param_groups = [
        {'params': fc_params, 'lr': learning_rate},
        {'params': pre_trained_params, 'lr': learning_rate*0.1},
    ]
    optimizer = torch.optim.Adam(param_groups)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
criterion = nn.CrossEntropyLoss()

best_model_state_dict = model.state_dict()
best_valid_acc = None
best_epoch = -1
df = pd.DataFrame(columns=['Epoch', 'Batch', 'Type', 'Loss', 'Accuracy'])
cur_index = 0

for epoch in range(num_epochs):
    model.train()
    total, correct, total_loss = 0, 0, 0
    
    with tqdm.tqdm(total=len(train_loader)) as pbar_train:
        for i, (modalities, labels, reps) in enumerate(train_loader):

            for modality, data in modalities.items():
                modalities[modality] = data.to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)
            reps = reps.to(device, non_blocking=True)

            outputs = model(modalities['pose_3d'],
                            modalities['eb_l_acc'], modalities['eb_l_gyr'],
                            modalities['sp_r_acc'], modalities['sp_r_gyr'],
                            modalities['sw_l_acc'], modalities['sw_l_gyr'],
                            modalities['sw_r_acc'], modalities['sw_r_gyr'])
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_loss_avg = total_loss / ((i + 1) * batch_size)
            total += labels.size(0)

            predicted = torch.argmax(outputs, dim=1)
            batch_correct = (predicted == labels).sum().item()
            correct += batch_correct
            acc = correct / total
            batch_acc = batch_correct / labels.size(0)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar_train.update(1)
            pbar_train.set_description('Epoch [{}/{}], Accuracy: {:.4f}, Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                                                              acc, total_loss_avg))

    df.loc[cur_index] = [epoch, None, 'train', total_loss_avg, acc]
    cur_index += 1
    
    if (epoch + 1) % args.checkpoint == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_valid_acc': best_valid_acc
        }, os.path.join(output_path, exp_name + '_checkpoint_' + str(epoch) + '.pth'))
    
    # Validation
    if (epoch + 1) % args.eval_every == 0:
        model.eval()
        with torch.no_grad():
            total, correct, total_loss = 0, 0, 0
            
            with tqdm.tqdm(total=len(val_loader)) as pbar_val:
                for i, (modalities, labels, reps) in enumerate(val_loader):

                    for modality, data in modalities.items():
                        modalities[modality] = data.to(device, non_blocking=True)

                    labels = labels.to(device, non_blocking=True)
                    reps = reps.to(device, non_blocking=True)

                    outputs = model(modalities['pose_3d'],
                                    modalities['eb_l_acc'], modalities['eb_l_gyr'],
                                    modalities['sp_r_acc'], modalities['sp_r_gyr'],
                                    modalities['sw_l_acc'], modalities['sw_l_gyr'],
                                    modalities['sw_r_acc'], modalities['sw_r_gyr'])

                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    total_loss_avg = total_loss / ((i + 1) * batch_size)
                    total += labels.size(0)

                    _, predicted = torch.max(outputs, dim=1)
                    batch_correct = (predicted == labels).sum().item()
                    correct += batch_correct
                    acc = correct / total
                    batch_acc = batch_correct / labels.size(0)

                    pbar_val.update(1)
                    pbar_val.set_description('Validation: Accuracy: {:.4f}, Loss: {:.4f}'.format(acc, total_loss_avg))

            if best_valid_acc is None or acc > best_valid_acc:
                best_valid_acc = acc
                steps_since_improvement = 0
                best_model_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            else:
                steps_since_improvement += 1
                if steps_since_improvement == args.early_stop:
                    df.loc[cur_index] = [epoch, None, 'validation', total_loss_avg, acc]
                    cur_index += 1
                    print('No improvement detected in the last %d epochs, exiting.' % args.early_stop)
                    break

        df.loc[cur_index] = [epoch, None, 'validation', total_loss_avg, acc]
        cur_index += 1
        print()
    
    scheduler.step(total_loss_avg)
    df.to_csv(os.path.join(output_path, exp_name + '.csv'), index=False)


################
# Evaluation
################

# Test best model
model.load_state_dict(best_model_state_dict)
with torch.no_grad():
    total, correct, total_loss = 0, 0, 0
    
    with tqdm.tqdm(total=len(test_loader)) as pbar_test:
        for i, (modalities, labels, reps) in enumerate(test_loader):

            for modality, data in modalities.items():
                modalities[modality] = data.to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)
            reps = reps.to(device, non_blocking=True)

            outputs = model(modalities['pose_3d'],
                            modalities['eb_l_acc'], modalities['eb_l_gyr'],
                            modalities['sp_r_acc'], modalities['sp_r_gyr'],
                            modalities['sw_l_acc'], modalities['sw_l_gyr'],
                            modalities['sw_r_acc'], modalities['sw_r_gyr'])
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_loss_avg = total_loss / ((i + 1) * batch_size)
            total += labels.size(0)

            _, predicted = torch.max(outputs, dim=1)
            batch_correct = (predicted == labels).sum().item()
            correct += batch_correct
            acc = correct / total
            batch_acc = batch_correct / labels.size(0)

            pbar_test.update(1)
            pbar_test.set_description('Test: Accuracy: {:.4f}, Loss: {:.4f}'.format(acc, total_loss_avg))

df.loc[cur_index] = [epoch, None, 'test', total_loss_avg, acc]
df.to_csv(os.path.join(output_path, exp_name + '.csv'), index=False)
torch.save({'model_state_dict': model.state_dict()},
           os.path.join(output_path, exp_name + '_e' + str(best_epoch) + '_best.pth'))
