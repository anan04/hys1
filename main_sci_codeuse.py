"""
通过导入星载的中心经纬度，自动查找机载数据的100*14m区段树高值
实现对比精度的功能

todo:
    在其他样地成功运行
    形成总的表格
    根据表格做分析，写文章
    


"""

# 外部函数
import gdal
import xlrd
import shapefile
from PIL import Image, ImageDraw
import re
import shutil
from osgeo import gdal
from osgeo import osr
import pandas as pd
from pandas import read_csv
import os
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
gdal.AllRegister()

from PIL import Image
# 自写函数
from latlon_batch import *
from latlon_inter import *


import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



def plot_mask(save_filename,img,mask,tifname,track,a,data3,index):
    """
    in : data
    out: img
    """
    
    os.makedirs(save_filename,exist_ok=1)
#     print(save_filename)
    plt.figure(16,figsize=(60,20))
    plt.title(tifname+track)
    
    # 机载图CHM
    plt.subplot(161)
    plt.imshow(img)
    # plt.imshow(mask)
    # 经过的星载区域（建立了缓冲区）
    plt.subplot(162)
    plt.imshow(mask)
    # plt.imshow(mask)
    # 机载像元转点显示
    plt.subplot(163)
    plt.plot(a,np.arange(len(a))[::-1],'r.')
#     plt.plot(a,np.arange(len(a))[::-1],'r.')
#     plt.xlabel(fontsize=20)

    # 统计信息
    plt.subplot(164)
 
    plt.plot([1,2],[2,2])
    try:
        per98 = str(round(np.percentile(a[a>0],[98])[0],2))
    except:
        per98 = '出现错误'
    try:
        per50 =str(round(np.percentile(a[a>0],[50])[0],2))
    except:
        per50 = '出现错误'
    try:
        gliht_max_tree = round(np.max(a[a>0]),2)
    except:
        gliht_max_tree = '出现错误'
 
    plt.text(1,2, '该区段内像元数=%s\n该区段内空值像元数占比=%s\n机载最大高=%s\n机载平均高(去0)=%s\n机载98高=%s\n机载50高=%s\n星载RH50=%s\n星载RH75=%s\nh_canopy(星载RH98)=%s'%(a.shape[0],
                                                                        len(a[np.isnan(a)])/len(a),
                                                                  
                                                              gliht_max_tree,
                                                                round(np.mean(a[a>0]),2),
                                                                                    per98,
                                                                                      per50,
                                                                                 round(data3.loc[index].RH50,2),
                                                                          round(data3.loc[index].RH75,2),
                                                                  round(data3.loc[index].h_canopy,2)

                                                     ),
             fontsize=24,va ='top', ha='left',  wrap=True,bbox=dict(boxstyle='round,pad=0.5',
                                                        fc='yellow', ec='k',lw=1 ))
    
    plt.subplot(165)
    
    a,b =   np.min(np.where(mask == 1)[0]),np.max(np.where(mask == 1)[0])
    c,d =   np.min(np.where(mask == 1)[1]),np.max(np.where(mask == 1)[1])
    plt.imshow(img[a:b,c:d])
    
    plt.subplot(166)
    alongTrack=(data3.loc[index].alongTrack)
    alongTracktemp=data3.alongTrack.apply(lambda x: pd.Series(fc(x)))
#     h_ph=(lambda x: pd.Series(fc(h_ph)))
    alongTrack=alongTracktemp.T[index]
    
    h_ph=(data3.loc[index].guanstudy)
    h_phtemp=data3.guanstudy.apply(lambda x: pd.Series(fc(x)))
#     h_ph=(lambda x: pd.Series(fc(h_ph)))
    h_ph=h_phtemp.T[index]
    plt.plot(h_ph,alongTrack,'r.')
    # try:
    #    plt.hist(a,200)
    # except:
    #    0
    # plt.show()
    plt.savefig(save_filename+'%s_%s_%s.jpg'%(tifname,track,index))
    plt.close()
    
    
def fc(x):
    a = re.findall(r"\d+\.?\d*",x)
    if a[1]=='38':
        return [-999 for  i in range(9)]
    else:
        return list(map(float, a))
def fcc(x):

    if ((x[1])=='-999'):
        return [-999 for  i in range(9)]

    else:
        return list(map(float, x))



def main(root_path,gliht,savedata,vis,gliht_rootpath="I:/huang/aiken/CHM/"):
    
    """
    in: gliht name
    out：result
    
    """
    
    zong_data =pd.DataFrame()

    data = pd.read_csv(root_path+'ATL03与GLIHT重叠点验证(atlas).csv')
    # 去掉空值数据
    data =  data[~data.canopy_h_metrics.isnull()].reset_index(drop=True)
    data[['RH'+str(i) for  i in [25,50,60,70,75,80,85,90,95]]] = data.canopy_h_metrics.apply(lambda x: pd.Series(fc(x)))
    import warnings; warnings.simplefilter('ignore')
    # 新建表
    new_data = pd.DataFrame()
    # data.filename ATL08文件
    for filename in np.unique(data.filename):
        print(filename)
        data2 = data[data.filename == filename].reset_index(drop=True)
        logger.info('正在处理:'+filename+'  '+str(data2.shape[0]))
        for tifname in np.unique(data2.tifname):
            print(tifname)
            data3 = data2[data2.tifname ==tifname].reset_index(drop=True)
            logger.info('--正在处理:'+tifname+'  '+str(data3.shape[0]))
    
            for track in np.unique(data3.track):
                
                data4 = data3[data3.track == track].reset_index(drop=True)
                
                
                logger.info('----正在处理:'+track+'  '+str(data4.shape[0]))
                if data4.shape[0]<3:
                    continue
                imgname = gliht_rootpath+tifname
                
                tifname_DTM = tifname.replace("CHM","DTM")
                imgname_DTM = "I:/huang/aiken/DTM/"+tifname_DTM
                
                print(imgname_DTM)
                print(track,'the number of cal data',data4.shape[0])
                mask_list = intersection(data4,imgname)
                
                mask_list_DTM = intersection(data4,imgname_DTM)
                
                logger.info('mask_list='+str(len(mask_list)))
    
                gdal_data = gdal.Open(imgname)
                pcs, gcs, extend, shape = get_file_info(gdal_data)
                img = gdal_data.GetRasterBand(1).ReadAsArray()
    
    
                gdal_data_DTM = gdal.Open(imgname_DTM)
                pcs_DTM, gcs_DTM, extend_DTM, shape_DTM = get_file_info(gdal_data_DTM)
                img_DTM = gdal_data_DTM.GetRasterBand(1).ReadAsArray()

                res = []
                res_gliht = []
                res_DTM = []
                res_gliht_DTM = []
                resglihtrh=[]
    
                # 是否绘图 参数
                filename = root_path+'overlay/mask_%s/'%gliht
    
                for index,mask in enumerate(mask_list):
    #                 print("mask=",mask.shape,len(mask[mask ==0]),len(mask))
    
                    a = img[mask.astype('bool')]
                    a_DTM = img_DTM[mask_list_DTM[index].astype('bool')]
                    

                    
                    logger.info("mask_list="+str(len(mask_list))+"---->>"+'len(a):'+str(len(a)))
    
                    # 把机载的相交的点保存起来
                    res_gliht.append(a)
                    res_gliht_DTM.append(a_DTM.tolist())
                    if (len(a) ==0):
                        res.append(-999)
                        resglihtrh.append([-999,-999,-999,-999,-999,-999,-999,-999,-999])
                    elif (len(a[np.isnan(a)])/len(a) > 0.2):
                        save_filename=filename+'mask-此区域空值超过0.8/'
                        if vis:plot_mask(save_filename,img,mask,tifname,track,a,data4,index)
                        res.append(-999)
                        resglihtrh.append([-999,-999,-999,-999,-999,-999,-999,-999,-999])
                    elif (a.shape[0]<100) :
                        save_filename=filename+'mask-点个数过少，可能只有半边/'
                        if vis:plot_mask(save_filename,img,mask,tifname,track,a,data4,index)
                        res.append(-999)
                        resglihtrh.append([-999,-999,-999,-999,-999,-999,-999,-999,-999])
                    elif (data4.loc[index].h_canopy > 9999):
                        save_filename=filename+'mask-星载数据在此区域值不正常/'
                        if vis:plot_mask(save_filename,img,mask,tifname,track,a,data4,index)
                        res.append(-999)
                        resglihtrh.append([-999,-999,-999,-999,-999,-999,-999,-999,-999])
                    elif ((a.shape[0]>100) &(len(a[np.isnan(a)])/len(a) < 0.2 )&(data4.loc[index].h_canopy < 9999)):
    #                     res.append(np.mean(a[a>0]))
                        try:
                            
                            "!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                            "把机载的多少高度拿去比较"
                            res.append(np.percentile(a[(a>0) & (a < 100)],[75]))
                            resglihtrh.append(np.percentile(a[a>0],[25,50,60,70,75,80,85,90,95]))
                        except:
                            res.append(-999)
                            resglihtrh.append(-[999,-999,-999,-999,-999,-999,-999,-999,-999])
                        save_filename=filename+'mask/'
                        if vis:plot_mask(save_filename,img,mask,tifname,track,a,data4,index)
    #                     res.append(np.percentile(a[a>0],[95]))
                    else:
                        save_filename=filename+'mask-其他/'
                        plot_mask(save_filename,img,mask,tifname,track,a,data4,index)
                        res.append(-999)
                        resglihtrh.append([-999,-999,-999,-999,-999,-999,-999,-999,-999])
                        #                    res_DTM = np.mean(a_DTM[~np.nan(a_DTM)])
                logger.info(str(len(res)))
                data4['new_tree'] = res
                data4['glihtrh'] = resglihtrh
                data4['gliht_point'] = res_gliht
                data4['gliht_point_DTM'] = res_gliht_DTM
                
                data4['new_land'] = 0
    
                new_data = pd.concat([new_data,data4])

    
    return new_data

def cal_acc(root_path,gliht,new_data):
    data = new_data
    zong_data =pd.DataFrame()
    data[['RH'+str(i) for  i in [25,50,60,70,75,80,85,90,95]]] = data.canopy_h_metrics.apply(lambda x: pd.Series(fc(x)))
    res = []
    data[['gliht_RH'+str(i) for  i in [25,50,60,70,75,80,85,90,95]]] = data.glihtrh.apply(lambda x: pd.Series(fcc(x)))
    col_gliht = 'new_tree'
    col_icesat = 'RH75'
    # mask =(data['new_tree']>0)
    # temp1=data[col_icesat]
    # temp2=data[col_gliht]
    # print(type(temp1),type(temp2))
    mask =(data[col_gliht]>0)&(abs(data[col_icesat]-data[col_gliht])<7)&(data[col_icesat]>0)
    # &(data['tifname']==tif)
    # &(data['track']=='gt2r')
    
    # 把所有符合条件的数据存起来
    if data[mask].shape[0]==0:
    #     return -999
        pass
    else:
        zong_data = pd.concat([zong_data,data])
    
    print('最终测评点数：',data[mask].shape[0])
    if data[mask].shape[0]==0:
        print('发成错误，没有重合点')
    os.makedirs(root_path+'overlay/mask_%s/'%(gliht),exist_ok=True)
    data.to_csv(root_path+'overlay/mask_%s/%s.csv'%(gliht,gliht),index=None)
    for  index,i in enumerate(['RH'+str(i) for  i in [25,50,60,70,75,80,85,90,95]]):
        col_icesat = i
        for index,i in enumerate(['gliht_RH'+str(i) for  i in [25,50,60,70,75,80,85,90,95]]):
            col_gliht=i
            print('---------------%s----------------'%col_icesat,col_gliht)
            string1=str(col_gliht+col_icesat)
            mse,rmse,rrmse,mae,r2,average,corrpearson,corrspearman,corrkendall,count,mean,std,mindata,fivetypercent,eightypercent,ninepercent,maxdata=accuracy_evaluation(data,mask,col_icesat,col_gliht,vis=0)
            res.append([string1,mse,rmse,rrmse,mae,r2,average,corrpearson,corrspearman,corrkendall,count,mean,std,mindata,fivetypercent,eightypercent,ninepercent,maxdata])
    res=pd.DataFrame(res)
    res.columns=['string1','mse','rmse','rrmse','mae','r2','average','corrpearson','corrspearman','corrkendall','count','mean','std','mindata','fivetypercent','eightypercent','ninepercent','maxdata']
    res.to_csv(root_path+'ATL03评价结果(atlas).csv',index=None)
        
        
        
    # res.append(accuracy_evaluation(data,mask,"new_land",vis=0))
    # res.append(accuracy_evaluation(data,mask,"h_te_median","new_land",vis=0))
    
    
    
if __name__ == "__main__":
    root_path = 'G:/东北林业大学/graduate202106/实验数据/aiken/2r2870000/'
    gliht='AMIGACarb_Augusta_FIA_Sep2011_l47s557_CHM(atlas).tif'
    savedata=1
    vis=1
    new_data = main(root_path,gliht,savedata,vis)
    np.save(root_path+'new_data.npy',new_data)
    cal_acc(root_path,gliht,new_data)
    print('完成区段评价')
    
    