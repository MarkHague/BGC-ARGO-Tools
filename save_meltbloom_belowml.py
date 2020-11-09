import numpy as np
import xarray as xr
import gsw
import pandas as pd
import glob, os
from scipy.signal import argrelextrema as lmm
import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns
from nan_helper import nan_helper

def save_meltbloom2(rtc = 0.1, bl = 0, DIR = "/Users/markhague/Google_Drive/ARGO/LOWRES"):
    # bl = number of levels to average below the estimated mixed layer
    # rtc =  realized degree of cooling over this winter surface layer - see Hague & Vichi, 2020, Biogeosciences
    # DIR = directory where float files are located, as well as where output will be saved 

    def find_nearest(array,value):
        idx = np.nanargmin(np.abs(array-value))
        return idx

    os.chdir(DIR)
    flist = glob.glob("*NQC.nc")
    print "No. of files:", len(flist)

    # arrays to hold results
    bloomdays  = np.array([])
    bloomweeks = np.array([])
    bloomweeks_m = np.array([])
    bloomyears = np.array([])
    meltweeks = np.array([])
    meltdays = np.array([])
    lons = np.array([])
    lats = np.array([])
    files = np.array([])
    len_ts = np.array([])
    peak_chl = np.array([])
    melt_week = np.array([])
    chl_sog = np.array([])
    mld_gi = np.array([])
    mld_chl_pos = np.array([])
    timing_pg = np.array([])
    grth_rate_gm = np.array([])
    gia_week = np.array([]) 

    for f in range(len(flist)):
        
        print "opening:", f
        ds = xr.open_dataset(flist[f])
        t = ds.Temperature
        t = t.where(t != -10000000000.0)
        t = t.where(ds.Temperature_QF == '0')
        s = ds.Salinity.where(ds.Salinity != -10000000000.0)
        s = s.where(s > 28.0)
        s = s.where(ds.Salinity_QF == '0')
        t = t.where(ds.Salinity > 28.0)
        pr = ds.Pressure.where(ds.Pressure != -10000000000.0)
        chl = ds.Chl_a.where(ds.Chl_a != -10000000000.0)
        chl = chl.where(ds.Chl_a_QF == '0')
        chl = np.where(chl > 0.0, chl, np.nan)
        lon = ds.Lon.where(ds.Lon != -10000000000.0)
        lat = ds.Lat.where(ds.Lat != -10000000000.0)
        
        #create list to hold results for each profile
        rl = [None] * t.shape[0]
        pi = np.zeros(t.shape[0])
        ts = np.array([])
        ss = np.array([])
        chls = np.array([])
        mlds = np.array([])
        
        # check for large discontinuities in time steps 
        # i.e if float was not operational for a long period
        if t.shape[0] >= 24: 
            time = pd.DatetimeIndex(ds.JULD.values)
            step = time[1:] - time[0:-1]
            tsmax = np.nanmax(step.days)
        
        if ((tsmax > 20) and (t.shape[0] > 48) and (chl.shape[0] > 48) ) or ( (tsmax < 20) and (t.shape[0] >= 24) and (chl.shape[0] >= 24) ): 

            for p in range(t.shape[0]):
                lonp = lon[p]
                latp = lat[p]
                pp = pr[p,:]
                chlp = chl[p,:]
                dp = gsw.z_from_p(pp, latp)
                if np.isnan(np.nanmean(dp)) == True:
                    dp = gsw.z_from_p(pp, lat.mean())
                try:
                    ic = (np.where(~np.isnan(chlp))[0]).max()
                    depi = dp[ic]
                except ValueError:
                    depi = -105.0
                i = np.nanargmin(pp)
                if depi > -100.0:
                    tp = t[p,:]
                    sp = s[p,:]
                    spa = gsw.SA_from_SP(sp, pp, lonp, latp)
                    if np.isnan(np.nanmean(spa)) == True:
                        spa = gsw.SA_from_SP(sp, pp, lon.mean(), lat.mean())
                    ct = gsw.CT_from_t(spa, tp, pp)
                    n2, pmid = gsw.Nsquared(spa, ct, pp)
                    dmid = gsw.z_from_p(pmid, latp)
                    if np.isnan(np.nanmean(dmid)) == True:
                        dmid = gsw.z_from_p(pmid, lat.mean())
                    #compute in-situ freezing temp based on salinity at surface
                    tfreeze = gsw.t_freezing(spa, 0.0)
                    # find index of minimum pressure - closest "non-nan/valid" observation to surface
                    try:
                        mld = dmid[np.nanargmax(n2)]
                        mlds = np.append(mlds, mld)
                        ddic = find_nearest(dp,mld)
                        # find index of deeper depth level (ddl) - say 10m below above observation
                        # in cases with coarse resolution sampling this will only be a single data point
                        ddl = dp[i] - 10.0
                        ddi = find_nearest(dp,ddl)
                        ts = np.append(ts,np.nanmean(tp[ddi:i])) 
                        ss = np.append(ss,np.nanmean(sp[ddi:i]))
                        chls = np.append(chls,np.nanmean(chlp[ddic-bl:ic]))
                    except ValueError:
                        ts = np.append(ts,np.nan) 
                        ss = np.append(ss,np.nan)
                        chls = np.append(chls,np.nan)
                        mlds = np.append(mlds, np.nan)

                else:
                    tp = t[p,:]*np.nan
                    tfreeze = t[p,:]*np.nan
                    ts = np.append(ts,np.nan) 
                    ss = np.append(ss,np.nan)
                    chls = np.append(chls,np.nan)
                    mlds = np.append(mlds,np.nan)

                #compare to observed temperature
                tdiff = tp[i] - tfreeze[i]
                if tdiff <= rtc:
                    rl[p] = 1 # under ice profile
                    pi[p] = 1 
                else:
                    rl[p] = 0

            count = rl.count(1)
            if (count >= 3): # at least 3 under ice profile and 8 months of data
                meltday = np.array([]) # list to hold results
                meltweek = np.array([])
                meltyear = np.array([])
                bloomday = np.array([])
                bloomweek = np.array([])
                bloomweek_m = np.array([])
                bloomyear = np.array([])
                blm_inds = np.array([])
                bloomlat = np.array([])
                bloomlon = np.array([])

                mld_a = np.array([])
                mld_fg = np.array([])
                pgweek = np.array([])

                dsalta = np.array([])
                dtempa = np.array([])
                grad_s = np.array([])
                grad_t = np.array([])
                mis = np.array([])
                gia_a = np.array([]) 
            
                week = time.weekofyear
                week = np.where(week <= 13, week+52, week)
                day = time.dayofyear
                day = np.where(day <= 89, day+365, day)
                year = time.year

                
                # fill-in missing values (e.g. no valid data within the ML)
                # this was manually checked to not be a problem - needs to be automated in some way
                # so the user knows how many consecutive nans are interpolated over. 
                if np.isnan(ts).any() == True:
                    nans, x = nan_helper(ts)
                    ts[nans]= np.interp(x(nans), x(~nans), ts[~nans])
                
                if np.isnan(ss).any() == True:
                    nans, x = nan_helper(ss)
                    ss[nans]= np.interp(x(nans), x(~nans), ss[~nans])
                
                if np.isnan(chls).any() == True:
                    nans, x = nan_helper(chls)
                    print nans
                    chls[nans]= np.interp(x(nans), x(~nans), chls[~nans])

                # smoothing of data is needed to remove high frequency (~10 days) variability
                # First, design the Buterworth filter
                N  = 1    # Filter order
                Wn = 0.1 # Cutoff frequency
                Wn2 = 0.2
                B, A = signal.butter(N, Wn, output='ba')
                B2, A2 = signal.butter(N, Wn2, output='ba')

                # Second, apply the filter
                # Some floats have extreme outliers, must filter these out first
                #chls = signal.medfilt(chls,5)
                chlsf = signal.filtfilt(B,A, chls)
                tsf = signal.filtfilt(B,A, ts)
                ssf = signal.filtfilt(B,A, ss)
                ssf2 = signal.filtfilt(B2,A2, ss[~np.isnan(ss)])

                #----------- COMPUTE TIMING OF MELT ONSET ------------------------------------------ #
                dt = tsf[1:] - tsf[0:-1]
                dst = ssf[1:] - ssf[0:-1]
                dst = np.where(dst < 0.0, dst, np.nan) # mask all increases in salinity
                #dm = dt*dst # if negative then temp is increasing while salinity decreases

                dpi = pi[1:] - pi[0:-1] # if = -1 then indicates switch from under ice to open ocean
                mi = np.where(dpi == -1)[0]

                #make sure time between switches (transitions of under ice to open water) is long enough to be seasonal i.e 3 months
                if len(mi) >= 1:
                    bi = np.array([])
                    for mp in enumerate(mi): 
                        nundi = len(np.where(pi[mp[1]-15:mp[1]] == 1)[0])
                        nmelt = mp[0]
                        if nundi < 2:
                            bi = np.append(bi, nmelt)
                    mi = np.delete(mi, bi)
                    
                    if len(mi) > 1:
                        # compute salinity derivative at potential melt points
                        dmi = np.abs(mi[1:] - mi[0:-1])
                        for nm, melts in enumerate(mi):
                            salt = ssf2[melts:melts+4] # salinity 1 month after potential melt
                            temp = tsf[melts:melts+4]
                            dsalt = np.nanmean(salt[1:] - salt[0:-1])
                            dtemp = np.nanmean(temp[1:] - temp[0:-1])

                            try:
                                if dmi[nm] > 12 and dsalt < 0.0 and nm <= len(dmi):
                                    mis = np.append(mis, melts)
                                    grad_s = np.append(grad_s, dsalt)
                                    grad_t = np.append(grad_t, dtemp)
                                    dsalt = np.nan
                            except IndexError:
                                dmi_t = dmi[-1]
                                if dmi_t > 12 and dsalt < 0.0:
                                    mis = np.append(mis, melts)
                                    grad_s = np.append(grad_s, dsalt)
                                    grad_t = np.append(grad_t, dtemp)
                                    dsalt = np.nan
                                    
                            dsalta = np.append(dsalta, dsalt)
                            dtempa = np.append(dtempa, dtemp)
                            
                        # cases where time between potential melts is sub-seasonal (> 90 days)
                        dsalta = np.where(dsalta < 0.0, dsalta, np.nan) # mask all increases in salinity
                        dmm = dsalta*dtempa
                        for nm, signals in enumerate(dsalta):
                            if signals < 0.0:
                                mis = np.append(mis, mi[nm])
                                grad_s = np.append(grad_s, signals)
                                grad_t = np.append(grad_t, dtempa[nm])
                                
                        # now check again for sub-seasonal consecutive melts, choose the one with
                        # the strongest signal
                        mis_inds = mis.argsort()
                        mis = np.sort(mis)
                        dmi_n = np.abs(mis[1:] - mis[0:-1])
                        inds_conf = np.where(dmi_n < 12)[0]
                        grad_s = np.where(grad_s < 0.0, grad_s, 0.0) # mask all increases in salinity
                        dmm = grad_s*grad_t
                        dmm = dmm[mis_inds]
                        
                        midel = np.array([])
                        for nm, index in enumerate(inds_conf):
                            signala = dmm[index+1]
                            if signala < dmm[index]:
                                midel = np.append(midel,index)
                            elif signala > dmm[index]:
                                midel = np.append(midel,index+1)
                                
                        mis = np.delete(mis, midel)
                    
                    elif len(mi) == 1:
                        mi = int(mi)
                        salt = ssf2[mi:mi+4] # salinity 1 month after potential melt
                        temp = tsf[mi:mi+4]
                        dsalt = np.nanmean(salt[1:] - salt[0:-1])
                        if dsalt < 0.0: 
                            mis = np.append(mis, mi)

                    for nm, m in enumerate(mis):
                        m = int(m)
                        meltday = np.append(meltday, day[m])
                        meltweek = np.append(meltweek, week[m])
                        melt_week = np.append(meltweek, week[m])
                        meltyear = np.append(meltyear, year[m])

                # ---------------------- compute chl growth initiation ----------------------------- #
                lmax = lmm(chlsf, np.greater)[0]
                lmin = lmm(chlsf, np.less)[0]
                nlmax, nlmin = len(lmax), len(lmin)
                dchlg = chls[1:] - chls[0:-1]
                chl_anom = chls - np.nanstd(chls)
                
                # some floats contain only nan values for chl, must filter these out
                if nlmin > 0 and nlmax > 0:
                    try:
                        dlmin = len(chlsf) - lmin[-1] # no. profiles after last local min
                    except IndexError:
                        dlmin, lmin = lmax[0], np.array([0])
                    try:
                        dlmax = lmax[0]
                    except IndexError:
                        dlmax, lmax = len(chlsf)-1, np.array([len(chlsf)-1])
                    nlmax, nlmin = len(lmax), len(lmin)
                    
                    if nlmin > nlmax and dlmin < 6:
                        lmin = np.delete(lmin,-1)
                    elif nlmin > nlmax and dlmin >= 6:
                        chlsm = np.copy(chls)
                        chlsm[0:lmin[-1]] = np.nan
                        lmaxe = np.nanargmax(chlsm)
                        lmax = np.append(lmax, lmaxe)
                    elif nlmin < nlmax and dlmax < 9:
                        lmax = np.delete(lmax, 0)
                    elif nlmin < nlmax and dlmax >= 9:
                        lmin = np.concatenate(([0], lmin))
                    elif nlmin == nlmax and dlmin >= 6:
                        dlmm = lmax - lmin
                        nneg = sum(n < 0 for n in dlmm)
                        if nneg > 0:
                            chlsm = np.copy(chls)
                            chlsm[0:lmin[-1]] = np.nan
                            lmaxe = np.nanargmax(chlsm)
                            lmax = np.append(lmax, lmaxe)
                            lmax = np.delete(lmax, 0)
                    elif nlmin == nlmax and dlmin < 6:
                        lmax = np.delete(lmax, 0)
                        lmin = np.delete(lmin, -1)
                             
                    for l in range(len(lmax)):
                        imax, imin = lmax[l], lmin[l]
                        imdiff = np.abs(imax-imin)
                        if (imin < imax) and (imdiff > 3):
                            chlmm = chls[imin:imax]
                            peak_chl = np.append(peak_chl, np.nanmax(chlmm))
                            weekmm = week[imin:imax]
                            daymm = day[imin:imax]
                            yearmm = year[imin:imax]
                            lonmm = lon[imin:imax]
                            for li,ll in enumerate(lonmm):
                                if ll > 315.0:
                                    lonmm[li] = lonmm[li] - 360.0 
                            latmm = lat[imin:imax]
                            # compute GI using alternative method of Tedesco et al. 2014
                            chl_anom_mm = chl_anom[imin:imax]
                            try:
                                gia_a = np.append( gia_a, weekmm[np.min(np.where(chl_anom_mm > 0))] )
                            except ValueError:
                                gia_a = np.append(gia_a, np.nan)
                            dchl = chlmm[1:] - chlmm[0:-1]
                            
                            #mask decreases in chl
                            dchl = np.where(dchl > 0.0, dchl, np.nan)
                            medchl = np.nanmedian(dchl)
                            blmind = np.nanargmax(dchl > medchl)
                            blmind_m = np.nanargmax(dchl)
                            chl_fi = np.nanargmax(dchl > 0.0)
                            chl_sog = np.append(chl_sog, chlmm[blmind])
                            blm_inds = np.append(blm_inds, np.min(np.where(dchlg == dchl[blmind])[0]))
                            bloomweek = np.append(bloomweek, weekmm[blmind])
                            bloomweek_m = np.append(bloomweek_m, weekmm[blmind_m])
                            bloomday = np.append(bloomday, daymm[blmind])
                            bloomyear = np.append(bloomyear, yearmm[blmind])

                            mldmm = mlds[imin:imax]
                            mld_a = np.append(mld_a, mldmm[blmind])
                            mld_fg = np.append(mld_fg, mldmm[chl_fi])
                            pgweek = np.append(pgweek, weekmm[chl_fi])

                            bloomlat = np.append(bloomlat, np.nanmean(latmm))
                            bloomlon = np.append(bloomlon, np.nanmean(lonmm))
                            if np.isnan(bloomlat.all()) == True or np.isnan(bloomlon.all()) == True:
                                bloomlat = np.append(bloomlat, np.nanmean(lat))
                                bloomlon = np.append(bloomlon, np.nanmean(lon))
                
    
                # filter out years with no melting event
                dlen = len(blm_inds) - len(mis)
                if dlen > 0 and len(mis) > 0:
                    mis_end = np.append(mis, np.ones(dlen)*mis[-1])
                    mis_begin = np.concatenate((np.ones(dlen)*mis[0], mis))
                    diff_e = np.abs(blm_inds - mis_end)
                    diff_b = np.abs(blm_inds - mis_begin)
                    teste = bloomweek[np.where(diff_e < 20)[0]]
                    if len(teste) == len(mis):
                        bloomweek = teste
                        bloomday = bloomday[np.where(diff_e < 20)[0]]
                        blm_inds = blm_inds[np.where(diff_e < 20)[0]]
                        bloomlat = bloomlat[np.where(diff_e < 20)[0]]
                        bloomweek_m = bloomweek_m[np.where(diff_e < 20)[0]]
                        bloomyear = bloomyear[np.where(diff_e < 20)[0]]
                        bloomlon = bloomlon[np.where(diff_e < 20)[0]]
                        mld_a = mld_a[np.where(diff_e < 20)[0]]
                        gia_a = gia_a[np.where(diff_e < 20)[0]]
                    elif len(teste) < len(mis):
                        testb = bloomweek[np.where(diff_b < 20)[0]]
                        if len(testb) == len(mis):
                            bloomweek = testb
                            bloomday = bloomday[np.where(diff_b < 20)[0]]
                            blm_inds = blm_inds[np.where(diff_b < 20)[0]]
                            bloomlat = bloomlat[np.where(diff_b < 20)[0]]
                            bloomweek_m = bloomweek_m[np.where(diff_b < 20)[0]]
                            bloomyear = bloomyear[np.where(diff_b < 20)[0]]
                            bloomlon = bloomlon[np.where(diff_b < 20)[0]]
                            mld_a = mld_a[np.where(diff_b < 20)[0]]
                            gia_a = gia_a[np.where(diff_b < 20)[0]]
                        else:
                            print 'Year with no melt not filtered'
                    

                len_ts = np.append(len_ts, len(chls))

                for e in range(len(meltday)):
                    if (len(bloomday) > 0) and (len(meltday) > 0):
                        try:
                            dd = np.abs(meltday[e] - bloomday[e])
                            if dd <= 90:
                                bloomdays = np.append(bloomdays, bloomday[e])
                                bloomweeks = np.append(bloomweeks, bloomweek[e])
                                bloomweeks_m = np.append(bloomweeks_m, bloomweek_m[e])
                                bloomyears = np.append(bloomyears, bloomyear[e])
                                meltweeks = np.append(meltweeks, meltweek[e])
                                meltdays = np.append(meltdays, meltday[e])
                                lons = np.append(lons, bloomlon[e])
                                lats = np.append(lats, bloomlat[e])
                                files = np.append(files, f)
                                mld_gi = np.append(mld_gi, mld_a[e])
                                mld_chl_pos = np.append(mld_chl_pos, mld_fg[e])
                                timing_pg = np.append(timing_pg, pgweek[e])
                                gia_week = np.append(gia_week, gia_a[e])
                                #compute the growth rate between GI and melting
                                diff_gm = mis[e] - blm_inds[e]
                                mis = mis.astype(np.int)
                                blm_inds = blm_inds.astype(np.int)
                                if diff_gm >= 2:
                                    chl_gm = chls[blm_inds[e]:mis[e]]
                                    dchl_gm = chl_gm[1:] - chl_gm[0:-1]
                                    dchl_gm = np.where(dchl_gm > 0.0, dchl_gm, np.nan)
                                    chl_gm_grate = np.nanmean(dchl_gm)
                                    grth_rate_gm = np.append(grth_rate_gm, chl_gm_grate)
                            elif dd > 90:
                                print 'Possible error, timing diff:', bloomweek[e]-meltweek[e]
                        except IndexError:
                            print "Warning: differing number of melts and blooms"

    print 'Avg. length of time series:', np.nanmean(len_ts)
    print 'Avg. peak chlorophyll:', np.nanmean(peak_chl)
    print 'Avg. melt week:', np.nanmean(melt_week)
    print 'Avg. chla at GI:', np.nanmean(chl_sog)
    print 'Avg. MLD at GI:', np.nanmean(mld_gi)
    print 'Avg. Growth Rate between GI and Melting:', np.nanmean(grth_rate_gm)
    
    print 'Avg. Timing of First Positive Chl:', np.nanmean(timing_pg)
    print 'Avg. MLD at First Positive Chl:', np.nanmean(mld_chl_pos)

    return bloomweeks, bloomdays, bloomweeks_m, bloomyears, meltweeks, meltdays, lons, lats, files, mld_gi, gia_week

bloomweeks, bloomdays, bloomweeks_m, bloomyears, meltweeks, meltdays, lons, lats, files, mld_gi, gia_week = save_meltbloom2(rtc = 0.1,bl = 0)
dsb = xr.Dataset({'bloomweeks': (['e'], bloomweeks),
                  'bloomdays': (['e'], bloomdays),
                  'bloomweeks_max': (['e'], bloomweeks_m),
                  'bloomyears': (['e'], bloomyears),
                  'meltweeks': (['e'], meltweeks),
                  'meltdays': (['e'], meltdays),
                  'GI_Ted2014': (['e'], gia_week),
                 'lons': (['e'], lons),
                 'lats': (['e'], lats),
                 'files': (['e'], files)})

#dsb.to_netcdf(DIR+'/meltbloom_gia.nc', format='NETCDF4_CLASSIC')
