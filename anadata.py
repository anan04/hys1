import spotpy
import numpy as np
import pandas as pd
def analydata(adata,gdata):
# 衡量线性回归的MSE 、 RMSE、 MAE
    data=pd.DataFrame()
    data['adata']=adata
    data['gdata']=gdata
    mse=[]
    rmse=[]
    rrmes=[]
    mae=[]
    r2=[]
    corrpearson=[]
    corrspearman=[]
    corrkendall=[]
    average=np.mean(gdata - adata)
    tempmse = np.sum((gdata - adata) ** 2) / len(adata)
    temprmse =spotpy.objectivefunctions.rmse(adata,gdata)
    tempmae = np.sum(np.absolute(adata - gdata)) / len(adata)
    tempr2 =spotpy.objectivefunctions.rsquared(adata,gdata)
    rrmse=spotpy.objectivefunctions.rrmse(adata,gdata)
    mse.append(tempmse)
    rmse.append(temprmse)
    mae.append(tempmae)
    r2.append(tempr2)
    tempcorrpearson=data['adata'].corr(data['gdata'],method='pearson')
    tempcorrspearman=data['adata'].corr(data['gdata'],method='spearman')
    tempcorrkendall=data['adata'].corr(data['gdata'],method='kendall')
    corrpearson.append(tempcorrpearson)
    corrspearman.append(tempcorrspearman)
    corrkendall.append(tempcorrkendall)
    diff=np.abs(data['adata']-data['gdata'])
    count,mean,std,mindata,fivetypercent,eightypercent,ninepercent,maxdata=diff.describe(include='all')
    print('complete acc')
    return mse,rmse,rrmse,mae,r2,average,corrpearson,corrspearman,corrkendall,count,mean,std,mindata,fivetypercent,eightypercent,ninepercent,maxdata
#计算树高
if __name__ == "__main__":
    image_save_path='G:/东北林业大学/graduate202106/实验数据/aiken/2r2870000/'
    path='G:/东北林业大学/graduate202106/实验数据/aiken/2r2870000/2r2870000.csv'
    h_phg=np.load(image_save_path+'h_phg.npy')
    h_phcwg=np.load(image_save_path+'h_phcwg.npy')
    achm=np.load(image_save_path+'achm.npy')
    hcanopy=np.load(image_save_path+'hcanopy.npy')
    hatlas=np.load(image_save_path+'hatlas.npy')
    atl03data= pd.read_csv(path, engine='python')
    signalmask=(atl03data['classification']>0)
    latitude=(atl03data['latitude'])[signalmask]
    lonitude=(atl03data['longitude'])[signalmask]
    signalmaskatlas=(atl03data['classification']>0)
    alongTrack=np.array((atl03data['alongTrack']))[signalmask]
    alongTrackatlas=np.array((atl03data['alongTrack'])[signalmaskatlas])
    h_ph=np.array(((atl03data['h_ph'])-(atl03data['geoid'])))[signalmask]
    h_phraw=h_ph
    h_phatlasraw=np.array(((atl03data['h_ph'])-(atl03data['geoid']))[signalmaskatlas])
    classification=np.array((atl03data['classification'])[signalmaskatlas])
    ATLASground=(classification==3)
    achm=np.array((atl03data['CHM'])[signalmask])   
    dis=max(alongTrack)-min(alongTrack)
    ATLASgroundlist=(classification==1)  
    ori_Y=h_ph
    guanceng= (ori_Y-h_phcwg)[((ori_Y-h_phcwg)>0.2) & (ori_Y<h_phg)]
    h_canopystudy = np.percentile(guanceng,[25,50,75,90,98])
    h_canopygliht=np.percentile(guanceng,[25,50,75,90,98])
    schm=h_phg-h_phcwg
    mse,rmse,rrmse,mae,r2,average,corrpearson,corrspearman,corrkendall,count,mean,std,mindata,fivetypercent,eightypercent,ninepercent,maxdata=analydata(achm,schm)
    print('schmmse',mse,'schmrmse',rmse,'schmrrmse',rrmse,'schmmae',mae,'schmr2',r2,'schmaverage',average,'schmcorrpearson',corrpearson,'schmcorrspearman',corrspearman,'schmcorrkendall',corrkendall,'schmcount',count,'schmmean',mean,'schmstd',std,'schmmindata',mindata,'schmfivetypercent',fivetypercent,'schmeightypercent',eightypercent,'schmninepercent',ninepercent,'schmmaxdata',maxdata)
    atlaschm=hcanopy-hatlas
    mse,rmse,rrmse,mae,r2,average,corrpearson,corrspearman,corrkendall,count,mean,std,mindata,fivetypercent,eightypercent,ninepercent,maxdata=analydata(achm,atlaschm)
    print('atlaschmmse',mse,'atlaschmrmse',rmse,'atlaschmrrmse',rrmse,'atlaschmmae',mae,'atlaschmr2',r2,'atlaschmaverage',average,'atlaschmcorrpearson',corrpearson,'atlaschmcorrspearman',corrspearman,'atlaschmcorrkendall',corrkendall,'atlaschmcount',count,'atlaschmmean',mean,'atlaschmstd',std,'atlaschmmindata',mindata,'atlaschmfivetypercent',fivetypercent,'atlaschmeightypercent',eightypercent,'atlaschmninepercent',ninepercent,'atlaschmmaxdata',maxdata)
