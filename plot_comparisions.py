#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:39:21 2017

@author: ajaver
"""

import os
import glob
import numpy as np
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

from read_features import FeatsReaderComp, FEATS_OW_MAP
from tierpsy.helper.misc import get_base_name

def plot_indv_feat(feats1, feats2, field, add_name=True, is_hist=False):
    xx = feats1[field]
    yy = feats2[field]
    
    tot = min(xx.size, yy.size)
    xx = xx[:tot]
    yy = yy[:tot]
    
    
    if is_hist:
        xx = xx[~np.isnan(xx)]
        yy = yy[~np.isnan(yy)]
        
        
        #get limits considering that one of the features is an empty list (all nan)
        if xx.size > 0:
            x_bot, x_top = np.min(xx), np.max(xx)
        else:
            x_bot, x_top = None, None
            
        if yy.size > 0:
            y_bot, y_top = np.min(yy), np.max(yy)
        else:
            y_bot, y_top = None, None
            
        bot = [x for x in (x_bot, y_bot) if x is not None]
        top = [x for x in (x_top, y_top) if x is not None]
        bot, top = (min(bot), max(top)) if len(bot)>=1 else (0,1)
        
        bins = np.linspace(bot, top, 100)
        
        count_x, _ = np.histogram(xx, bins)
        count_y, _ = np.histogram(yy, bins)
        
        l1 = plt.plot(bins[:-1], count_x)
        l2 = plt.plot(bins[:-1], count_y)
        if add_name:
            my = max(np.max(count_x), np.max(count_y))*0.95
            mx = bins[1]
            plt.text(mx, my, field)
            #plt.title(field)
        return (l1, l2)
    else:
        ll = plt.plot(xx, yy, '.', label=field)
        ran1 = plt.ylim()
        ran2 = plt.xlim()
        
        ran_l = ran1 if np.diff(ran1) < np.diff(ran2) else ran2
        
        plt.plot(ran_l, ran_l, 'k--')
        if add_name:
            my = (ran1[1]-ran1[0])*0.97 + ran1[0]
            mx = (ran2[1]-ran2[0])*0.03 + ran2[0]
            plt.text(mx, my, field)
        
        return ll
        #plt.legend(handles=ll, loc="lower right", fancybox=True)
        #plt.axis('equal')

def _get_common_feats(feats1, feats2):
    valid_feats = set(feats1.keys()) & set(feats2.keys())
    
    valid_feats = [x for x in valid_feats 
                   if isinstance(feats1[x], np.ndarray) and 
                   isinstance(feats2[x], np.ndarray)]
    
    valid_feats = [x for x in valid_feats  
                     if feats1[x].size > 1 and
                     feats2[x].size > 1]
    
    return valid_feats


def plot_feats_comp(feats1, feats2, add_name=True, is_hist=False):
    valid_feats = _get_common_feats(feats1, feats2)
    
    tot_f1 = max(feats1[x].size for x in valid_feats)
    tot_f2 = max(feats2[x].size for x in valid_feats)
    tot = min(tot_f1, tot_f2)
    
    ii = 0
    
    sub1, sub2 = 5, 6
    tot_sub = sub1*sub2
    
    all_figs = []
    for field in sorted(valid_feats):
        if feats1[field].size == 1 or feats2[field].size == 1:
            continue
        
        if is_hist and \
        not (feats1[field].size >= tot and feats2[field].size >= tot):
            continue
        
            
        if ii % tot_sub == 0:
            fig = plt.figure(figsize=(14,12))
            all_figs.append(fig)
            
        sub_ind = ii%tot_sub + 1
        ii += 1
        plt.subplot(sub1, sub2, sub_ind)
        
        
        plot_indv_feat(feats1, feats2, field, add_name, is_hist)
        
    
    return all_figs
#%%
def save_features_pdf(tierpsy_feats, 
                      segworm_feats, 
                      pdf_file,
                      feats2plot=None,
                      xlabel='tierpsy features',
                      ylabel='segworm features'):
    rev_dict = {val:key for key,val in FEATS_OW_MAP.items()}
    
    if feats2plot is None:
        feats2plot = _get_common_feats(tierpsy_feats, segworm_feats)
    
    if not any('.' in x for x in feats2plot):
        feats2plot = [FEATS_OW_MAP[x] for x in feats2plot]
            
    feats2plot = sorted(feats2plot)
    with PdfPages(pdf_file) as pdf_id:
        for ow_field in feats2plot:
            #if not 'locomotion.crawling_bends' in ow_field:
            #    continue 
            field = rev_dict[ow_field]
            
            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plot_indv_feat(tierpsy_feats, segworm_feats, field, add_name=False, is_hist=False)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            
            
            plt.subplot(1,2,2)
            ax = plot_indv_feat(tierpsy_feats, segworm_feats, field, add_name=False, is_hist=True)
            L=plt.legend(ax)
            for ii, ff in enumerate((xlabel, ylabel)):
                L.get_texts()[ii].set_text(ff)
            
            plt.suptitle(ow_field)
            pdf_id.savefig()
            plt.close()
            
            plt.figure(figsize=(10,5))
            ax = plt.subplot(1,1,1)
            
            x1 = tierpsy_feats[field]
            x1 = x1[:min(x1.size, 1000)]
            ax2= plt.plot(x1)
            
            x2 = segworm_feats[field]
            x2 = x2[:min(x2.size, 1000)]
            ax1 = plt.plot(x2)
            
            plt.title(ow_field)
            L=plt.legend((ax1, ax2), loc=1)
            for ii, ff in enumerate((xlabel, ylabel)):
                L.get_texts()[ii].set_text(ff)
            pdf_id.savefig()
            plt.close()
            
            
    return tierpsy_feats, segworm_feats

if __name__ == '__main__':
    
    save_plot_dir = os.path.join('.', 'plots')
    if not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)
    
    main_dir = '/data/ajaver/onedrive/Ev_videos/example/'
    feat_files = glob.glob(os.path.join(main_dir, '**', '*_features.hdf5'), recursive=True)
    feats2plot = None
    for feat_file in feat_files:
        print(feat_file)
        segworm_feat_file = feat_file.replace('.hdf5', '.mat')
        basename = get_base_name(feat_file)
        
        pdf_file = os.path.join(save_plot_dir, basename + '_feat_comparison.pdf')
        feats_reader = FeatsReaderComp(feat_file, segworm_feat_file)
        tierpsy_feats = feats_reader.read_plate_features()
        segworm_feats = feats_reader.read_feats_segworm()
        
        
        
        tierpsy_feats, segworm_feats = \
        save_features_pdf(tierpsy_feats, 
                          segworm_feats, 
                          pdf_file,
                          feats2plot=feats2plot)
        
        


