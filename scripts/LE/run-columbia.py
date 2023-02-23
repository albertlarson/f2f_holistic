import f2f
import torch
import numpy as np
import hydroeval as he
import delorean
import pandas as pd
import itertools as it

torch.cuda.empty_cache()

sst = [True,False]
zscorex = [False, True]
shuffled = [False, True]
tts = [.70,.80,.90,.95]
lag = [0,1,3,6,12,24]
basin = ['Columbia']
nn_hid_lay_size = [4,30,200,1000,'dcrrnn']

lr = 1e-4
epochs = 100
batch_size = 2
configs = []
for x in it.product(sst,zscorex,shuffled,tts,lag,basin,nn_hid_lay_size):
    configs.append(x)
configs = np.asarray(configs)
configs.shape

for IDX,X in enumerate(configs):
    torch.cuda.empty_cache()
    sst = X[0]
    zscorex = X[1]
    shuffled = X[2]
    tts = float(X[3])
    lag = int(X[4])
    basin = X[5]
    if X[6] == 'dcrrnn':
        nn_hid_lay_size = X[6]
    else:
        nn_hid_lay_size = int(X[6])
    clippedim=False
    if basin == 'Columbia':
        sf = torch.load('../../data/traintest/COL_STFL_traintest.pt')
    elif basin == 'Yukon':
        sf = torch.load('../../data/traintest/YUK_STFL_traintest.pt')
        sf = sf.to(torch.float32)
        
    if (sst == 'True') and (basin == 'Columbia') and (zscorex == 'True'):
        imz = torch.load('../../data/traintest/COL_MOGL_ZSCORE_traintest.pt')
    elif (sst == 'True') and (basin == 'Yukon') and (zscorex == 'True'):
        imz = torch.load('../../data/traintest/YUK_MOGL_ZSCORE_traintest.pt')
    elif (sst == 'True') and (basin == 'Columbia') and (zscorex == 'False'):
        imz = torch.load('../../data/traintest/COL_MOGL_traintest.pt')
    elif (sst == 'True') and (basin == 'Yukon') and (zscorex == 'False'):
        imz = torch.load('../../data/traintest/YUK_MOGL_traintest.pt')
    elif (sst == 'False') and (basin == 'Columbia'):
        imz = torch.load('../../data/traintest/COL_CLIP_traintest.pt')
        imz = f2f.fill_NOwhiten(imz)
        clippedim = True
    elif (sst == 'False') and (basin == 'Yukon'):
        imz = torch.load('../../data/traintest/YUK_CLIP_traintest.pt')
        imz = f2f.fill_NOwhiten(imz)
        clippedim = True
    dset = f2f.dset_maker(imz,sf,imz.shape[0],lag,zscorex=zscorex,clippedim=clippedim)
    # cube_height = dset.x.shape[2]
    # cube_width = dset.x.shape[3]
    train_size = int(tts*len(dset))
    test_size = int((len(dset) - train_size))
    train_dset = torch.utils.data.TensorDataset(dset.x[:train_size],dset.y[:train_size])
    test_dset = torch.utils.data.TensorDataset(dset.x[train_size:],dset.y[train_size:])
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dset, batch_size=batch_size, shuffle=True)
    if nn_hid_lay_size != 'dcrrnn':
        model = f2f.a_linear(dset.x.shape[1] * dset.x.shape[2] * dset.x.shape[3],dset.y.shape[-1],intermed_layer_size=nn_hid_lay_size)
    else:
        model = f2f.a(dset.x.shape[1] * dset.x.shape[2] * dset.x.shape[3],dset.y.shape[-1],chanz=dset.x.shape[1])
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = torch.nn.MSELoss()
    t0 = delorean.Delorean().shift('US/Eastern').datetime
    for i in range(epochs):
        model.train()
        for idx,(x,y) in enumerate(train_dataloader):
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            predicted = model(x)   
            loss = criterion(predicted.reshape(-1),y.reshape(-1))
            loss.backward()
            optimizer.step()
    t1 = delorean.Delorean().shift('US/Eastern').datetime
    t2 = t1 - t0
    t2 = t2.seconds / 60
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dset, batch_size=1, shuffle=False)
    test_pred = torch.empty((0,test_dset.tensors[-1].shape[-1])).to('cuda')
    test_yy = torch.empty((0,1,1,test_dset.tensors[-1].shape[-1])).to('cuda')
    with torch.no_grad():
        for xx,yy in test_dataloader:
            pred = model(xx.cuda())
            test_pred = torch.cat((test_pred,pred))
            test_yy = torch.cat((test_yy,yy.cuda()))
    nse_test = he.evaluator(he.nse,test_pred.cpu().detach().numpy().reshape(-1,1),test_yy.cpu().detach().numpy().reshape(-1,1)) 
    predicts_correct_within_twosigma_of_self = []
    count_correct_within_twosigma = []
    for i in range(test_pred.shape[-1]):
        j = test_pred.cpu()[:,i]
        k = j + (2 * torch.std(j))
        l = j - (2 * torch.std(j))
        m = test_dset.tensors[-1].cpu().squeeze()[:,i]
        n = np.where((m<=k) & (m>=l),1,0)
        n1 = np.sum(n)
        o = round(100*np.sum(n)/n.shape[0],2)
        count_correct_within_twosigma.append(n1)
        predicts_correct_within_twosigma_of_self.append(o)
    avg_count = np.format_float_positional(np.mean(count_correct_within_twosigma),precision=2)
    avg_acc = np.format_float_positional(np.mean(predicts_correct_within_twosigma_of_self),precision=2)
    input_params = {
    'sst':[X[0]],
    'zscorex':[X[1]],
    'shuffled':[X[2]],
    'tts':[X[3]],
    'lag':[X[4]],
    'basin':[X[5]],
    'nn_hid_lay_size':[X[6]],
    }
    output_params = {
        'input_shape':[imz.shape],
        'output_shape':[sf.shape],
        'train_len':[len(train_dset)],
        'train_time_min':[round(t2,3)],
        'test_len':[len(test_dset)],
        'nse_test':[np.format_float_positional(nse_test[0],3)],
        'avg_count':[avg_count],
        'avg_acc':[avg_acc],
        'gpu':[torch.cuda.get_device_name(0)]
    }
    parms = {**input_params,**output_params}
    if IDX == 0:
        df = pd.DataFrame.from_dict(parms)
        df.to_pickle('../../data/results/pickles/fin-columbia.pkl')
    else:
        df = pd.read_pickle('../../data/results/pickles/fin-columbia.pkl')
        df1 = pd.DataFrame.from_dict(parms)
        df = pd.concat([df,df1],ignore_index=True)
        df.to_pickle('../../data/results/pickles/fin-columbia.pkl')
    torch.save(model.state_dict(),f'../../data/results/models/columbia/{IDX}.ptm')
#     if IDX == 4:
#         break
        
# df
