#!/usr/bin/python

import sys
import numpy as np
from numpy import linalg as LA
from numpy import NaN, Inf, arange, isscalar, asarray, array
from string import atof
from string import replace

au2ev=27.211385

n_roots = int(sys.argv[1])
#istate=int(sys.argv[3])
n_states_excl=np.size(sys.argv)-3
states_excl=[]
for i in range (0,n_states_excl):
	states_excl.append(int(sys.argv[i+3]))

states_excl=np.sort(states_excl)
def main(argv):

    data = open(argv[2],'r')

   


    # Default: include CAP only trough the first order of PT

    mode = '11100'

    #if len(argv) == 4:
    #    mode = str(argv[3])

    current_line = data.readline()

    captions = ['*** HEFF(0+1) REAL PART ***',
            '*** HEFF(2) REAL PART ***',
            '*** WEFF(1) IMAG PART ***',
            '*** WEFF(2) REAL PART ***',
            '*** WEFF(2) IMAG PART ***']

    heff_parts = []

    for dummy in range(len(captions)):
        heff_parts.append([])

# Do I need to specify the termination line?
    term = 71*'-' 
# Number of coloms per line
    num_of_cols = 5 if n_roots > 5 else n_roots
# The length of the last chunk 
    last_seg = n_roots%num_of_cols if n_roots > num_of_cols else n_roots

    while current_line != '':
        
        if current_line.strip() in captions:
            collected = False
            idx = captions.index(current_line.strip())

            for dummy in range(2):
                current_line = data.readline()

            while not collected:
                current_line = data.readline()
                if not term in current_line:
                    for elem in current_line.strip().split()[1:]:
                        heff_parts[idx].append(atof(replace(elem,'D','E')))
                else:
                    if len(heff_parts[idx]) == n_roots**2:
                        collected = True
                    else:
                        current_line = data.readline()
                        # script makes extra turn if the separator
                        # is encountered
                        continue

        current_line = data.readline()

#    print 'heff_parts', heff_parts

# Identify segments and assemble matrices
    
    heff_parts_matr = []

    for h_i in heff_parts:
#        print 'h_i', h_i
        lst_of_matr = []
        num_of_chunks = int(n_roots/num_of_cols)
        if last_seg != n_roots: 
            num_of_chunks += 1

        for dummy in range(num_of_chunks - 1):
            tmp = np.array(h_i[0:(n_roots*num_of_cols)]).reshape(n_roots,num_of_cols)
            lst_of_matr.append(tmp)
            del h_i[0:(n_roots*num_of_cols)]
# According to Python rules the last index of the slice is excluded
#        print 'h_i last segment', h_i
        tmp = np.array(h_i[0:(n_roots*last_seg)]).reshape(n_roots,last_seg)
        lst_of_matr.append(tmp)

        heff_parts_matr.append(np.concatenate(lst_of_matr, axis = 1))


    xmc_diag_fesh(heff_parts_matr)
#    print heff_parts_matr

# Generate the trajectory and save it to files

    eta0 = 1e-5
# N2
#    eta_step = 0.00001 up to #3_3
    eta_step = 0.0005
#    eta_step = 0.000005 
#    eta_step = 0.00002 
#    eta_step = 0.0001 
#    eta_step = 0.001 
#    eta_step = 0.00005 
#    eta_step = 0.0001 up to #3
#    eta_step = 0.00001
# H2
#    eta_step = 0.25
#    eta_step = 0.05
#    eta_step = 0.005
#    eta_step = 0.0001
    stats_for_states = [17]
# BQN
#    eta_step = 0.5
#    eta_step = 1.0
#    eta_step = 0.05
#    eta_step = 0.25
#    eta_step = 0.001
#    eta_step = 0.01
#    eta_step = 0.01
#    eta_step = 0.00025
#    eta_step = 0.1
#    eta_step = 0.0005
#    eta_step = 0.005

#Photoswitch

#    eta_step = 0.002
#    eta_step = 0.00025
#    nsteps = 300
#    nsteps = 1000
    nsteps = 2000

    print "-eta_step = %10.6f\n-nsteps = %d" % (eta_step, nsteps)


    traj_e = []
    traj_u = []
    traj_v = []
    traj_e_raw = []
    traj_u_raw = []
    traj_v_raw = []
    for dummy in range(n_roots):
        traj_e.append([])
        traj_u.append([])
        traj_v.append([])
        traj_e_raw.append([])
        traj_u_raw.append([])
        traj_v_raw.append([])

    eta_vals = [ eta0 + i*eta_step for i in range(nsteps) ]

    for eta in eta_vals:
        w, v, u = xmc_diag(heff_parts_matr, eta, mode)
        w = w.tolist()
        for en in w:
            root = w.index(en)
            traj_e_raw[root].append((en.real,2*en.imag))
            traj_u_raw[root].append((u[root].real,2*u[root].imag))
            traj_v_raw[root].append(v[:,root])

    # When done building the trajectory perform sorting

    # Initialize

    for root in range(n_roots):
        traj_e[root].append(traj_e_raw[root][0])
        traj_u[root].append(traj_u_raw[root][0])
        traj_v[root].append(traj_v_raw[root][0])

    # Sort using sort_thresh

    #sort_thresh = 10
    sort_thresh = 1e-3


    for step in range(1, nsteps):

        for root in range(n_roots):

            nearest_within_thresh = -1
            min_dist = np.inf
            min_dist_v = np.inf

            e, g = traj_e[root][-1]
            eig_vec = np.array(traj_v[root][-1])

            for root_c in range(n_roots):
                e_c, g_c = traj_e_raw[root_c][step]
                eig_vec_c = np.array(traj_v_raw[root_c][step])

                d = np.sqrt((e_c - e)**2 + (g_c - g)**2)
                dv = eig_vec - eig_vec_c
                dv_norm = np.sqrt(np.dot(dv.conj(), dv))

                if d < min_dist and d < sort_thresh and dv_norm < min_dist_v:
                    min_dist = d
                    min_dist_v = dv_norm
                    nearest_within_thresh = root_c

            if nearest_within_thresh == -1:
                # Copy previous point
                traj_e[root].append(traj_e[root][-1])
                traj_u[root].append(traj_u[root][-1])
                traj_v[root].append(traj_v[root][-1])
            else:
                traj_e[root].append(traj_e_raw[nearest_within_thresh][step])
                traj_u[root].append(traj_u_raw[nearest_within_thresh][step])
                traj_v[root].append(traj_v_raw[nearest_within_thresh][step])




    for state in range(n_roots):
        traj_file = open('traj_'+str(state+1)+'.dat', 'w')
        trajc_file = open('trajc_'+str(state+1)+'.dat', 'w')
        traj_d_file = open('traj_d_'+str(state+1)+'.dat', 'w')
        traj_dc_file = open('traj_dc_'+str(state+1)+'.dat', 'w')
        eta_e_file = open('eta_e_'+str(state+1)+'.dat', 'w')
        eta_g_file = open('eta_g_'+str(state+1)+'.dat', 'w')
        eta_ec_file = open('eta_ec_'+str(state+1)+'.dat', 'w')
        eta_gc_file = open('eta_gc_'+str(state+1)+'.dat', 'w')

        eig_str_files = []
        for cas_vec in range(n_roots):
            eig_str_files.append(open('eig_'+str(state+1)+'_cas'+str(cas_vec+1)+'.dat', 'w'))

        # coefficients (abs) of casscf vectors are arranged along rows:
        eig_info = np.concatenate(traj_v[state])
        eig_info = np.reshape(eig_info, (n_roots, nsteps), order='F')

        # restructure energy data for diff_traj subroutine
        traj_e_matr = np.array(traj_e[state])
        traj_u_matr = np.array(traj_u[state])
        traj_d = diff_traj(eta_vals, traj_e_matr[:,0], traj_e_matr[:,1], 'normal')
        traj_dc = diff_traj(eta_vals, traj_u_matr[:,0], traj_u_matr[:,1], 'corrected')

#        print traj_d

        for i in range(len(eta_vals)):
#            print i
            traj_file.write(str(traj_e[state][i][0]) + ' ' + str(traj_e[state][i][1]) + '\n')
            trajc_file.write(str(traj_u[state][i][0]) + ' ' + str(traj_u[state][i][1]) + '\n')

            if i <= len(eta_vals) - 3:
                traj_d_file.write(str(traj_d[i][0]) + ' ' + str(traj_d[i][1]) + '\n')
                traj_dc_file.write(str(traj_dc[i][0]) + ' ' + str(traj_dc[i][1]) + ' ' + str(traj_dc[i][2]) +'\n')

            eta_e_file.write(str(eta_vals[i]) + ' ' + str(traj_e[state][i][0]) + '\n')
            eta_g_file.write(str(eta_vals[i]) + ' ' + str(traj_e[state][i][1]) + '\n')
            eta_ec_file.write(str(eta_vals[i]) + ' ' + str(traj_u[state][i][0]) + '\n')
            eta_gc_file.write(str(eta_vals[i]) + ' ' + str(traj_u[state][i][1]) + '\n')
            for cas_vec in range(n_roots):
                eig_str_files[cas_vec].write(str(eta_vals[i]) + ' ' + str(eig_info[cas_vec,i]) + '\n')

        traj_file.close()
        trajc_file.close()
        eta_e_file.close()
        eta_g_file.close()
        eta_ec_file.close()
        eta_gc_file.close()

        for f in eig_str_files:
            f.close()

        if state in stats_for_states:
            maxtab, mintab = peakdet(np.array(traj_d)[:,1], 1e-10)
            maxtab_e, mintab_e = peakdet(traj_e_matr[:,0], 1e-10)
            maxtab_g, mintab_g = peakdet(traj_e_matr[:,1], 1e-10)
#            print mintab
            num_of_minima = mintab.shape[0]
            num_of_minima_e = mintab_e.shape[0]
            num_of_maxima_e = maxtab_e.shape[0]
            num_of_minima_g = mintab_g.shape[0]
            num_of_maxima_g = maxtab_g.shape[0]
            # NOTE: indexes in traj_d are shifted due to the way diff_traj works
            # in order to recover "original" position 1 is added to the actual index
            # obtained from mintab[:,0]
            print "================Uncorrected trajectory================"
            print ">>> Results of trajectory analysis for state %d <<<" % (state + 1)
            print ">>> Local minima of log velocity" 
            print "-- eta -- -- velocity -- -- E (a.u.) -- -- Gamma (a.u.) --"
            for i in range(num_of_minima):
                e, g = traj_e[state][int(mintab[i,0])+1] # one is added to account for the shift in traj_diff
                print "%9.5f %14.8f %14.8f %18.10f" % (eta_vals[int(mintab[i,0])+1],mintab[i,1], e, g)

            print ">>> Stationary points on E(eta) and -G(eta)" 
            print ">>> Minima" 
            print "-- eta -- -- E (a.u.) --"
            for i in range(num_of_minima_e):
                e, g = traj_e[state][int(mintab_e[i,0])]
                print "%9.5f %14.8f " % (eta_vals[int(mintab_e[i,0])], e)
            print "-- eta -- -- (-1)*G (a.u.) --"
            for i in range(num_of_minima_g):
                e, g = traj_e[state][int(mintab_g[i,0])]
                print "%9.5f %14.8f " % (eta_vals[int(mintab_g[i,0])], g)
            print ">>> Maxima" 
            print "-- eta -- -- E (a.u.) --"
            for i in range(num_of_maxima_e):
                e, g = traj_e[state][int(maxtab_e[i,0])]
                print "%9.5f %14.8f " % (eta_vals[int(maxtab_e[i,0])], e)
            print "-- eta -- -- (-1)*G (a.u.) --"
            for i in range(num_of_maxima_g):
                e, g = traj_e[state][int(maxtab_g[i,0])]
                print "%9.5f %14.8f " % (eta_vals[int(maxtab_g[i,0])], g)

            maxtab_dure, mintab_dure = peakdet(np.array(traj_dc)[:,1], 1e-10)
            maxtab_duim, mintab_duim = peakdet(np.array(traj_dc)[:,2], 1e-10)
            maxtab_e, mintab_e = peakdet(traj_u_matr[:,0], 1e-10)
            maxtab_g, mintab_g = peakdet(traj_u_matr[:,1], 1e-10)
#            print mintab
            num_of_minima_re = mintab_dure.shape[0]
            num_of_minima_im = mintab_duim.shape[0]
            num_of_minima_e = mintab_e.shape[0]
            num_of_maxima_e = maxtab_e.shape[0]
            num_of_minima_g = mintab_g.shape[0]
            num_of_maxima_g = maxtab_g.shape[0]
            print "================ Corrected trajectory ================"
            print ">>> Results of trajectory analysis for state %d <<<" % (state + 1)
            print ">>> Local minima of dU_re/deta" 
            print "-- eta -- -- derivative -- -- E (a.u.) --"
            for i in range(num_of_minima_re):
                e, g = traj_u[state][int(mintab_dure[i,0]) + 1]
                print "%9.5f %14.8f %14.8f" % (eta_vals[int(mintab_dure[i,0]) + 1],mintab_dure[i,1], e)

            print ">>> Local minima of dU_im/deta" 
            print "-- eta -- -- derivative -- -- Gamma (a.u.) --"
            for i in range(num_of_minima_im):
                e, g = traj_u[state][int(mintab_duim[i,0]) + 1]
                print "%9.5f %14.8f %18.10f" % (eta_vals[int(mintab_duim[i,0]) + 1],mintab_duim[i,1], g)

            print ">>> Stationary points on E(eta) and -G(eta)" 
            print ">>> Minima" 
            print "-- eta -- -- E (a.u.) --"
            for i in range(num_of_minima_e):
                e, g = traj_u[state][int(mintab_e[i,0])]
                print "%9.5f %14.8f " % (eta_vals[int(mintab_e[i,0])], e)
            print "-- eta -- -- (-1)*G (a.u.) --"
            for i in range(num_of_minima_g):
                e, g = traj_u[state][int(mintab_g[i,0])]
                print "%9.5f %14.8f " % (eta_vals[int(mintab_g[i,0])], g)
            print ">>> Maxima" 
            print "-- eta -- -- E (a.u.) --"
            for i in range(num_of_maxima_e):
                e, g = traj_u[state][int(maxtab_e[i,0])]
                print "%9.5f %14.8f " % (eta_vals[int(maxtab_e[i,0])], e)
            print "-- eta -- -- (-1)*G (a.u.) --"
            for i in range(num_of_maxima_g):
                e, g = traj_u[state][int(maxtab_g[i,0])]
                print "%9.5f %14.8f " % (eta_vals[int(maxtab_g[i,0])], g)

def xmc_diag_fesh(heff_i):
    heff = heff_i[0] + heff_i[1]
    cap_fo = 0.5 * (heff_i[2] + heff_i[2].transpose()) # first order CAP
    W_mat=cap_fo
    heff=0.5*(heff+heff.transpose())
    heff_tmp=heff
 
    #Diagonalize H_eff matrix and reorder states
    xmcq_e,xmcq_v=LA.eig(heff_tmp)
    xmcq_v,R_mat=LA.qr(xmcq_v)
    heff_tmp=np.dot(xmcq_v.transpose(),np.dot(heff_tmp,xmcq_v))
    print "Diagonal Heff"
    print heff_tmp
    new_order=np.argsort(xmcq_e)
    eigenvalues=xmcq_e[new_order]
    ref_en=np.min(eigenvalues)
    print "-- Diagonal matrix in adiabatic basis -- "
    print "State        Energy"
    for i in range (0,n_roots):

	print "%d       %10.6f" % (i+1, (au2ev*(eigenvalues[i]-ref_en)))
    	 
    for j in range (0,n_roots):
	xmcq_v[j,:]=xmcq_v[j,new_order]

   #Transform CAP matrix to the basis that diagonalizes Heff
    cap_xmcq=np.dot(xmcq_v.transpose(),np.dot(W_mat,xmcq_v))
    cap_xmcq=0.5*(cap_xmcq+cap_xmcq.transpose())
    heff_tmp=np.dot(xmcq_v.transpose(),np.dot(heff,xmcq_v))
    heff_tmp=0.5*(heff_tmp+heff_tmp.transpose())
    print "Diagonal Heff matrix"
    print heff_tmp
    ref_en=np.min(np.diagonal(heff_tmp))

    for i in range (0,n_states_excl):
	j=states_excl[i]-1
	cap_xmcq=np.delete(cap_xmcq,j-i,0)
	cap_xmcq=np.delete(cap_xmcq,j-i,1)
	heff_tmp=np.delete(heff_tmp,j-i,0)
	heff_tmp=np.delete(heff_tmp,j-i,1)

	
    w_diab, v_diab = LA.eig(cap_xmcq)

    heff_re_diab = np.dot(v_diab.transpose(), np.dot(heff_tmp, v_diab))
    heff_re_diab = 0.5 * (heff_re_diab + heff_re_diab.transpose())

    n_states = heff_re_diab.shape[0]
    print "-- Resonance parameters in diabatic basis --"
    print "-- state #          E, a.u.     Res Shift, a.u.        G, a.u.  --"


    tmp=np.diagonal(heff_re_diab)

    for k in range (0,n_states):
	heff_istate=heff_re_diab
	heff_istate=np.delete(heff_istate,k,0)
	heff_istate=np.delete(heff_istate,k,1)
	diab_diag_e,diab_diag_v=LA.eig(heff_istate)
	diab_diab_v,R_mat=LA.qr(diab_diag_v)
	diab_diag_v=np.insert(diab_diag_v,k,0,0)
	diab_diag_v=np.insert(diab_diag_v,k,0,1)
	diab_diag_v[k,k]=1.0

	heff_istate=np.dot(diab_diag_v.transpose(),np.dot(heff_re_diab,diab_diag_v))
 	heff_istate=0.5*(heff_istate+heff_istate.transpose())	


    	Gamma = 0.0
    	Res_shift=0.0
    	for j in range(0,k)+range(k+1, n_states):
			if abs(heff_istate[k,k]-heff_istate[j,j])>0.00001:
            			Gamma += heff_istate[k, j]**2
	    			Res_shift += heff_istate[k,j]**2/(heff_istate[k,k]-heff_istate[j,j])
    	Gamma *= 2 * np.pi


    	print "%d     %13.6f   %13.6f   %13.6f " % (k + 1, au2ev*(heff_istate[k, k]-ref_en), au2ev*(heff_istate[k, k]-ref_en)+au2ev*Res_shift, au2ev*Gamma)
#    print "-- Diabatic states - Eigenstates of CAP matrix --\n"	
#    for i in range (n_states-n_states_excl):
#	print "State #",i+1
#	st_order=np.argsort(abs(v_diab[:,i]))
#	v_diab_sort=v_diab[st_order,i]
#	for k in range (n_states):
#		print v_diab_sort[n_states-n_states_excl-k-1],st_order[n_states-n_states_excl-k-1]+1,
#	print "\n"

def xmc_diag(heff_i, eta, mode='11100',theta=0.0):


# mode is a bitmask with positions corresponding
# to the contributions to effective Hamiltonian
# defalt 11111 (h(0+1)_re, h(2)_re, weff(1)_im, weff(2)_re, weff(2)_im)

    b = int(mode, 2)
    eta = eta*np.exp(1.0j*theta*np.pi/180.0) 

    heff_re = ((b>>4)%2)*heff_i[0] + ((b>>3)%2)*heff_i[1] - ((b>>1)%2)*heff_i[3]*eta**2
    cap_fo = ((b>>2)%2)*heff_i[2] # first order CAP
#    heff_im = -eta*(((b>>2)%2)*heff_i[2] + (b%2)*heff_i[4])
    heff_im = -eta*(cap_fo + (b%2)*heff_i[4])
    heff = heff_re + 1.0j*heff_im
#
    heff = 0.5*(heff + heff.transpose())
#
    w, v = LA.eig(heff)

# As for now we are interested in trajectories only 
# eigenvectors will be discarded

#    w = w.tolist()
#    w.sort(key=lambda x: x.real) 

    re_w = w.real
    sorted_idx = np.argsort(re_w) # Now w[sorted_idx] <--> v[:,sorted_idx]

    sorted_w = w[sorted_idx]
    sorted_v = abs(v[:,sorted_idx])
    sorted_v_c = v[:,sorted_idx]

    # Compute corrected trajectories through the first order of PT
    # Symmetrize cap_fo just in case

    cap_fo = 0.5*(cap_fo + cap_fo.transpose())

    num_of_roots = sorted_w.shape[0]
    corr_traj = [ 0.0 + 1.0j*0.0 for dummy in range(num_of_roots) ]
    norm_sq = 0.0 + 1.0j*0.0
    cap_av = 0.0 + 1.0j*0.0

    for i in range(num_of_roots):
        norm_sq = np.dot(sorted_v_c[:,i], sorted_v_c[:,i])
#        print "Root %d |x|**2 = %12.6f + j %12.6f" % (i, norm_sq.real, norm_sq.imag)
        cap_av = np.dot(sorted_v_c[:,i], np.dot(cap_fo,sorted_v_c[:,i]))
#        print cap_av
        corr_traj[i] = sorted_w[i] + 1.0j*eta*cap_av/norm_sq

    corr_w = np.array(corr_traj)

#    print "corr_w: "
#    print corr_w

    return (sorted_w, sorted_v, corr_w) 


def diff_traj(x, y1, y2, mode):
    traj_d = []

    for i in range(1,len(x)-1):
        d1 = (y1[i+1] - y1[i-1])/(x[i+1] - x[i-1])
        d2 = (y2[i+1] - y2[i-1])/(x[i+1] - x[i-1])
        if mode == 'normal':
            traj_d.append((x[i], x[i]*np.sqrt(d1**2 + d2**2)))
        elif mode == 'corrected':
            traj_d.append((x[i], np.sqrt(d1**2), np.sqrt(d2**2)))

    return traj_d

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

if __name__ == '__main__':
    main(sys.argv)
