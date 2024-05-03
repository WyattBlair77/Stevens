import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Literal
from sklearn.metrics import r2_score, mean_squared_error
import os
import pandas as pd

from data import get_data


# ========= Unpack data so utils can reference it ========= 

data_dict = get_data()

train, test = data_dict['train_df'], data_dict['test_df']
train_subset = data_dict['train_subset'], data_dict['test_subset']
train_loader, train_loader_shuffle, test_loader = data_dict['train_loader'], data_dict['train_loader_shuffle'], data_dict['test_loader']

sequence_length = data_dict['sequence_length']

# ========================================================= 

def plot_data(ax, alpha=1):

    ax.set_xlabel('Date')
    ax.set_ylabel('Points')
    ax.set_title('Industrial Production: Utilities: Electric and Gas Utilities Index (IPG2211A2N)')

    ax.plot(train['POINTS'], label='train', color='blue', alpha=alpha)
    ax.plot(test['POINTS'], label='test', color='red', alpha=alpha)

    ax.xaxis.set_major_locator(plt.MaxNLocator(18))
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid()

def evaluate_model(model, num_epochs, lr, model_type=None, **train_kwargs):

    if model_type is None:
        # optimization
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # train
        loss_log = train_model(
            model=model, 
            criterion=criterion, 
            optimizer=optimizer, 
            num_epochs=num_epochs,
        )

        # test
        test_model(model=model, loss_log=loss_log)

    elif model_type == 'gan':

        loss_log = train_gan_model(num_epochs=num_epochs, **train_kwargs)
        test_gan_model(loss_log=loss_log, **train_kwargs)

def train_gan_model(generator, discriminator, num_epochs, d_lr, g_lr):

    # Optimizers
    optim_g = optim.Adam(generator.parameters(), lr=g_lr)
    optim_d = optim.Adam(discriminator.parameters(), lr=d_lr)

    # Loss function
    criterion = nn.BCELoss()
    
    losses = {}

    # Training
    for epoch in tqdm(range(num_epochs), desc=f'Training GAN', total=num_epochs):
        for date_index, y_true in train_loader_shuffle:

            date_index = date_index[0].float()
            y_true = y_true[0].float()[:, np.newaxis]

            # generate fake data
            noise = torch.randn(train_loader.batch_size, generator.input_dim)
            fake_data = generator(date_index*noise)

            # train discriminator
            optim_d.zero_grad()
            pred_real = discriminator(y_true)
            loss_real = criterion(pred_real, torch.ones_like(pred_real))
            pred_fake = discriminator(fake_data.detach())
            loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            optim_d.step()

            # train generator
            optim_g.zero_grad()
            pred_fake = discriminator(fake_data)
            loss_g = criterion(pred_fake, torch.ones_like(pred_fake))
            loss_g.backward()
            optim_g.step()

            losses.update({epoch: {'g': loss_g, 'd': loss_d}})
    
    return losses

def test_gan_model(loss_log, **kwargs):

    model_name = 'GAN'

    generator = kwargs['generator']
    discriminator = kwargs['discriminator']

    generator.eval()
    discriminator.eval()

    fig, (pred_ax, loss_ax) = plt.subplots(2, 1, figsize=(7, 10))
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    
    train_pred = predict_full_series(generator, train_loader, sequence_length=sequence_length, train_or_test='train')
    test_pred = predict_full_series(generator, test_loader, sequence_length=sequence_length, train_or_test='test')

    model_metrics = get_metrics(train_pred, test_pred)
    save_metrics(model_name, model_metrics)

    pred_ax.plot(train.index, train_pred.detach(), label='train-prediction', color='orange', linestyle='--', linewidth=5, alpha=1)
    pred_ax.plot(test.index , test_pred.detach() , label='test-prediction' , color='green' , linestyle='--', linewidth=5, alpha=1)
    
    plot_data(pred_ax, alpha=0.5)

    pred_ax.set_title(f"{model_name} Prediction")

    d_loss = [v['d'].detach() for v in loss_log.values()]
    g_loss = [v['g'].detach() for v in loss_log.values()]

    loss_ax.plot(loss_log.keys(), d_loss, label='discriminator loss')
    loss_ax.plot(loss_log.keys(), g_loss, label='generator loss')

    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss')
    loss_ax.set_title(f'{model_name} Loss')

    fig.legend(loc='upper right')

    plt.savefig(f'./model_evaluations/{model_name}.png')
    plt.show()
    
    return model_metrics

def train_model(model, criterion, optimizer, num_epochs):
    
    losses = {}

    # train network
    for epoch in tqdm(range(num_epochs), desc=f'Training {model.__class__.__name__}', total=num_epochs):

        for date_index, y_true in train_loader_shuffle:

            date_index = date_index[0].float()
            y_true = y_true[0].float()[:, np.newaxis]

            # forward
            outputs = model(date_index).float()
            loss = criterion(outputs, y_true)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.update({epoch: loss})

    return losses

def test_model(model, loss_log, test_pred_adjustment_func=None):

    model_name = model.__class__.__name__
    model.eval()

    fig, (pred_ax, loss_ax) = plt.subplots(2, 1, figsize=(7, 10))
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    
    train_pred = predict_full_series(model, train_loader, sequence_length=sequence_length, train_or_test='train')
    test_pred =  predict_full_series(model, test_loader , sequence_length=sequence_length, train_or_test='test' )    

    model_metrics = get_metrics(train_pred, test_pred)
    save_metrics(model_name, model_metrics)

    if test_pred_adjustment_func is not None: test_pred = test_pred_adjustment_func(test_pred)

    pred_ax.plot(train.index, train_pred.detach(), label='train-prediction', color='orange', linestyle='--', linewidth=5, alpha=1)
    pred_ax.plot(test.index , test_pred.detach() , label='test-prediction' , color='green' , linestyle='--', linewidth=5, alpha=1)
    
    plot_data(pred_ax, alpha=0.5)

    pred_ax.set_title(f"{model_name} Prediction")

    loss_ax.plot(loss_log.keys(), [v.detach() for v in loss_log.values()])
    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss')
    loss_ax.set_title(f'{model_name} Loss')

    fig.legend(loc='upper right')

    try:
        plt.savefig(f'./model_evaluations/{model_name}.png')
    except:       
        plt.savefig(f'../model_evaluations/{model_name}.png')
    plt.show()

    return model_metrics

def predict_full_series(model, dataloader, sequence_length, train_or_test: Literal['train', 'test']):
    
    model.eval()  # Set the model to evaluation mode
    df = data_dict[f'{train_or_test}_df']

    predictions = torch.zeros(len(df.index))  # Full series length
    counts = torch.zeros(len(df.index))  # Count overlaps
    
    with torch.no_grad():

        for i, (x_sequence, _) in enumerate(dataloader):

            output = model(x_sequence[0])  # Predict
            output = output.squeeze().detach()  # Remove batch dimension and detach from graph
            
            predictions[i:i + sequence_length] += output  # Add predictions to their positions
            counts[     i:i + sequence_length] += 1       # Increment counts for averaging

    counts[counts == 0] += 1 # avoid divide by zero issues
    predictions /= counts  # Average predictions
    return predictions

def plot_metrics():

    model_scores = pd.read_csv('./model_scores.csv', index_col=0)

    fig, (r2_ax, mse_ax) = plt.subplots(2,1, figsize=(10, 10), sharex=True)

    r2_keys = ['train_r2_score', 'test_r2_score', 'full_r2_score']
    mse_keys = ['train_mean_squared_error', 'test_mean_squared_error', 'full_mean_squared_error']

    model_scores[r2_keys].plot.bar(ax=r2_ax)
    model_scores[mse_keys].plot.bar(ax=mse_ax)

    mse_ax.set_title('Mean Squared Error')
    mse_ax.grid()
    mse_ax.set_xticklabels(model_scores.index)

    r2_ax.set_title('$R^2$ Score')
    r2_ax.grid()

    fig.suptitle('Model Metric Comparison')

    plt.savefig('./model_scores.png')
    plt.show()

def get_metrics(train_pred, test_pred):

    full_pred = np.append(train_pred, test_pred)
    
    train_r2_score = r2_score(train['POINTS']                , train_pred)
    test_r2_score  = r2_score(test[ 'POINTS']                , test_pred )
    full_r2_score =  r2_score(data_dict['fft_data']['POINTS'], full_pred )

    train_mean_squared_error = mean_squared_error(train['POINTS']                , train_pred)
    test_mean_squared_error  = mean_squared_error(test[ 'POINTS']                , test_pred )
    full_mean_squared_error =  mean_squared_error(data_dict['fft_data']['POINTS'], full_pred )

    output_metrics = {
        'train_r2_score': train_r2_score,
        'test_r2_score': test_r2_score,
        'full_r2_score': full_r2_score,

        'train_mean_squared_error': train_mean_squared_error,
        'test_mean_squared_error': test_mean_squared_error,
        'full_mean_squared_error': full_mean_squared_error,
    }

    print(f'R2-Scores: ')
    print(f'--> TRAIN: %1.5f' % train_r2_score)
    print(f'--> TEST : %1.5f' % test_r2_score )
    print(f'--> FULL : %1.5f' % full_r2_score )

    print(f'MSE-Scores: ')
    print(f'--> TRAIN: %1.5f' % train_mean_squared_error)
    print(f'--> TEST : %1.5f' % test_mean_squared_error )
    print(f'--> FULL : %1.5f' % full_mean_squared_error )

    return output_metrics

def save_metrics(model_name, metric_dict):

    current_dir = os.getcwd()
    
    if current_dir.endswith('models'):
        csv_fp = '../model_scores.csv'
    else:
        csv_fp = './model_scores.csv'

    if os.path.exists(csv_fp):
        score_df = pd.read_csv(csv_fp, index_col=0)

        row_df = pd.DataFrame(columns=metric_dict.keys(), index=pd.Index([model_name]))
        row_df[list(metric_dict.keys())] = list(metric_dict.values())

        score_df = pd.concat([score_df, row_df])
    
    else:

        score_df = pd.DataFrame(columns=metric_dict.keys(), index=pd.Index([model_name]))
        score_df[list(metric_dict.keys())] = list(metric_dict.values())

    score_df.to_csv(csv_fp)