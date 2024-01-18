import numpy as np
import copy, subprocess, os, yaml, sys, math

from random import randint
import matplotlib.animation as animation
from numpy.lib.type_check import nan_to_num
import initial, boundary, cfxx, rhs, hcal, newgrd, cip1d
import qbz,phisval,qsucal,shear,source,eddyvis,diffusion, posread

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
#from matplotlib._version import get_versions as mplv

def nays1d_Mini(case_name):
    
    conf_file=case_name[0:6]+'.yml'
    bed_file=case_name[0:3]+'_xzb.csv'
#    print(conf_file)

    with open(conf_file,'r', encoding='utf-8') as yml:   
#        config = yaml.load(yml)
        config = yaml.safe_load(yml)
    xl=float(config['xl']); nx=int(config['nx'])
    j_param=int(config['j_param'])
    j_channel=int(config['j_channel'])
    slope=float(config['slope'])
    x_slope=float(config['x_slope'])
    slope1=float(config['slope1']); slope2=float(config['slope2'])
    xb1=float(config['xb1']); xb2=float(config['xb2']); xb3=float(config['xb3'])
    dbed=float(config['dbed'])
    xw1=float(config['xw1']); xw2=float(config['xw2']); xw3=float(config['xw3']); xw4=float(config['xw4'])
    w1=float(config['w1']);w2=float(config['w2']); w3=float(config['w3'])

    qp=float(config['qp']); g=float(config['g']); snm=float(config['snm'])
    alh=float(config['alh']); lmax=int(config['lmax']); errmax=float(config['errmax'])
    hmin=float(config['hmin'])
    j_upstm=int(config['j_upstm']); j_dwstm=int(config['j_dwstm'])
    j_dwstm_cnd=int(config['j_dwstm_cnd']); h_downstream=float(config['h_downstream'])
    j_ini_prof=int(config['j_ini_prof'])
    etime=float(config['etime']); dt=float(config['dt']); tuk=float(config['tuk'])
    b_start=float(config['b_start']); o_start=float(config['o_start'])
    alpha_up=float(config['alpha_up']); alpha_dw=float(config['alpha_dw'])
    alpha_up_sed=float(config['alpha_up_sed'])

    diam=float(config['diam']);sub=float(config['sub']);lam=float(config['lam'])
    nu=float(config['nu']); kappa=float(config['kappa'])
    musk=float(config['musk'])
    j_sediment=int(config['j_sediment']);j_sus_ini=int(config['j_sus_ini'])

    xsize=float(config['xsize'])
    ymin=float(config['ymin']);ymax=float(config['ymax'])
    umin=float(config['umin']);umax=float(config['umax'])
    tausta_min=float(config['tausta_min']);tausta_max=float(config['tausta_max'])
    qmin=float(config['qmin']);qmax=float(config['qmax'])
    frmin=float(config['frmin']);frmax=float(config['frmax'])
    qbmin=float(config['qbmin']);qbmax=float(config['qbmax'])

    sgd=sub*g*diam; sgd3=sgd*diam*diam
    sgd3_s=math.sqrt(sgd3); sgd_s=math.sqrt(sgd);sd=sub*diam

    wf=phisval.fallvel(sgd3,sgd_s,nu)
    tsc=phisval.iwgk(diam,sub,g)
    gamma_1=math.sqrt(tsc/musk)

    if j_param==2:
        j_channel=2
        npos,xpos,zpos,bpos,slope1,slope2=posread.pos(bed_file)
        slp=slope1
    else:
        if j_channel==1:
            slp=slope
        else:
            slp=slope1

    hs0_use=(snm*qp/(w1*math.sqrt(slp)))**(3./5.)
    usta0=math.sqrt(g*hs0_use*slp)
    uswf=usta0/wf
    if uswf<1.08:
        kwd='bedload'
    elif uswf<1.67:
        kwd='bedload+suspended load'
    else:
        kwd='suspended load'

    nx1=nx+1; nx2=nx+2
    dx=xl/nx; xct=xl/2.
    rlam=dt/(1.-lam)

    x=np.linspace(0,xl,nx+1)
    xmid=(x[0]+x[nx])*.5; nxm=int(nx/2)
    x_cell=np.zeros([nx2])
    x_cell=initial.x_cell_init(x_cell,x,dx,nx)
    it_out=int(tuk/dt)

    # Make Array
    hs=np.zeros([nx2]); hs_up=np.zeros([nx2]); fr=np.zeros([nx2])
    h=np.zeros([nx2]); hn=np.zeros([nx2]); h_up=np.zeros([nx2])
    u=np.zeros([nx2]); un=np.zeros([nx2])
    eta=np.zeros([nx2]); eta0=np.zeros([nx2]); eta_up=np.zeros([nx2]);eta_up0=np.zeros([nx2])
    deta=np.zeros([nx2])
    cfx=np.zeros([nx2]); qu=np.zeros([nx2])
    gux=np.zeros([nx2]); gux_n=np.zeros([nx2])
    fn=np.zeros([nx2]);gxn=np.zeros([nx2])
    h0_up=np.zeros([nx2]); hc_up=np.zeros([nx2]); hs0_up=np.zeros([nx2]); hce_up=np.zeros([nx2])
    bslope=np.zeros([nx2]); bslope_up=np.zeros([nx2])
    b=np.zeros([nx2]); b_up=np.zeros([nx2]); u0_up=np.zeros([nx2])

    c=np.zeros([nx2]); cn=np.zeros([nx2]); cb=np.zeros([nx2]); gcx=np.zeros([nx2])
    wfcb=np.zeros([nx2])
    ie=np.zeros([nx2]);tausta=np.zeros([nx2]);qb=np.zeros([nx2]);qbt=np.zeros([nx2])
    qsu=np.zeros([nx2]); qsu_x=np.zeros([nx2])
    usta=np.zeros([nx2]);usta_x=np.zeros([nx2])
    alpha=np.zeros([nx2]); beta=np.zeros([nx2])
    epsilon=np.zeros([nx2]); dcdx=np.zeros([nx2])

    # Geometric Condition
    if j_param==1:
        if j_channel==1:
            eta,eta0,eta_up,eta_up0=initial.eta_init \
                (eta,eta0,eta_up,eta_up0,nx,dx, \
                slope,xl,xb1,xb2,xb3,dbed)
        else:
            eta,eta0,eta_up,eta_up0=initial.eta_init_2 \
                (eta,eta0,eta_up,eta_up0,nx,dx, \
                xl,x_slope,slope1,slope2)
    else:
        eta,eta0,eta_up,eta_up0=initial.eta_init_r(eta,eta0,eta_up,eta_up0,nx,x,x_cell,npos,xpos,zpos)

    if j_param==1:
        b,b_up=initial.width_init(b,b_up,nx,dx,xw1,xw2,xw3,xw4,xl,w1,w2,w3)
        bslope,bslope_up=initial.bslope_cal(nx,dx,bslope,bslope_up\
        ,j_channel,slope,x_slope,slope1,slope2)
    else:
        b_b_up=initial.width_init_r(b,b_up,nx,x,x_cell,npos,xpos,bpos)
        bslope,bslope_up=initial.bslope_cal_r(nx,dx,bslope,bslope_up,eta,eta_up,slope1,slope2)

    # Uniform Flow Depth and Critical Depth
    hs0_up,h0_up,u0_up=initial.h0_cal(nx,dx,qp,snm,eta_up, \
        hs0_up,h0_up,b_up,bslope_up,u0_up)    
    hc_up,hce_up=initial.hc_cal(hc_up,hce_up,eta_up,qp,nx,b_up,g)

    # 初期河床と川幅縦断図

    ysize=xsize*.5
    fig,ax=plt.subplots(figsize = (xsize, ysize))

    #河床のプロット(左軸)
    xmin=x[0];xmax=x[nx]
    xlen=xmax-xmin; xmin=xmin-xlen*.05; xmax=xmax+xlen*.05
    im=ax.set_xlim(xmin,xmax)
    im=ax.set_ylim(ymin,ymax)
    im=ax.set_title("Longitudinal Proile",size='30')
    im=ax.set_xlabel("x(m)",size='25')
    im=ax.set_ylabel("Elevation(m)",size='25')
    for tick in ax.get_xticklabels():
        tick.set(fontsize=25)
    for tick in ax.get_yticklabels():
        tick.set(fontsize=25)

    #im=ax.plot(x[0:nx],eta_up[0:nx],label='Bed Elevation',linewidth=4,color='black')
    im=ax.plot(x_cell[0:nx+2],eta[0:nx+2],label='Bed Elevation-1',linewidth=4,color='black')
    ax.text(0.,eta_up[0],"Bed Elevation",size=30,color='black')

    #im=ax.plot(x[0:nx+1],h0_up[0:nx+1],label='Uniform WSE',linewidth=4,color='green')
    #ax.text(0.,h0_up[0],"Uniform WSE",size=30,color='green')

    #im=ax.plot(x[0:nx+1],hce_up[0:nx+1],label='Critical WSE',linewidth=4,color='purple')
    #ax.text(0.,hce_up[0],"Critical WSE",size=30,color='purple')

    #幅のプロット(右軸)
    ax1=ax.twinx() #右軸
    bmax=np.max(b_up)*1.2; bmin=0.
    im1=ax1.set_ylim(bmin,bmax)
    im1=ax1.set_ylabel("Channel Width(m)",size='25')
    im1=ax1.plot(x[0:nx],b_up[0:nx],label='Channel Width',linewidth=4,color='red')
    for tick in ax1.get_yticklabels():
        tick.set(fontsize=25)
    ax1.text(x[nx]*.8,b_up[nx],"Channel Width",size=30,color='red')

    #fig.savefig("ini_longitudinal_prof.png")
    #plt.clf(); plt.close()

    # Initial Depth and Water Surface Elevation
    hs_upstm=hs0_up[0]*alpha_up ; h_upstm=eta[0]+hs_upstm
    if j_dwstm_cnd ==1:
        hs_dwstm=hs0_up[nx]*alpha_dw
    elif j_dwstm_cnd ==2:
        hs_dwstm=h_downstream 

    h_dwstm=eta[nx+1]+hs_dwstm

    h,hs,h_up,hs_up=initial.h_init \
            (eta,eta_up,h,hs,h_up,hs_up,hs0_up,h_upstm,h_dwstm,hs_upstm,hs_dwstm,nx,dx,xl,j_ini_prof)

    # Hydraulic and Physical Parameters
    u_upstm=qp/(b_up[0]*hs_upstm)
    u_dwstm=qp/(b_up[nx]*hs_dwstm)

    h,hs=boundary.h_bound(h,hs,eta,h_upstm,h_dwstm,hs_dwstm,nx,j_upstm,j_dwstm)

    hs_up=boundary.hs_up_cal(hs,hs_up,nx,hs_upstm,hs_dwstm,j_upstm,j_dwstm)
    h_up=boundary.h_up_cal(hs_up,eta_up,h_up,nx)
    hn=copy.copy(h)

    u,fr=initial.u_init(g,qp,u,qu,hs_up,b_up,fr,nx)
    u=boundary.u_bound(u,hs_up,qp,b_up,nx,j_upstm,j_dwstm,u_upstm,u_dwstm)
    un=copy.copy(u)

    #初期水位のプロット
    im=ax.plot(x_cell[0:nx+1],h[0:nx+1],label='Initial WSE',linewidth=4,color='blue')
    #im=ax.plot(x[0:nx],h_up[0:nx],label='Initial WSE',linewidth=4,color='green')
    ax.text(x_cell[1],h[1],"Initial WSE",size=30,color='blue')

    fig.savefig("ini_longitudinal_prof.png")
    plt.clf(); plt.close()

    # Initial ie and cfx
    cfx,ie=cfxx.cfx_cal(cfx,nx,un,hs_up,ie,g,snm)

    # Initial usta and tausta
    usta,tausta,fr=shear.usts(u,usta,tausta,fr,ie,hs_up,g,sd,nx)
    usta,tausta,fr=boundary.shear_bound(usta,tausta,fr)
    usta_x=boundary.usta_xcal(usta,usta_x,nx)

    # Initial alpha and beta
    alpha,beta=phisval.alpha_beta(alpha,beta,usta_x,wf,nx)

    # Initial qb  
    if j_sediment==1 or j_sediment==3:
        qb=qbz.qbcal(qb,tausta,tsc,sgd3_s,nx,eta,dx,gamma_1)
        qb=boundary.qb_bound(qb,alpha_up_sed)
        qbt=qb*b_up

    # Initial qsu
    if j_sediment>=2:
        qsu=qsucal.itakura(qsu,tausta,sgd_s,wf,nx)
        qsu=boundary.qsu_bound(qsu)
        qsu_x=boundary.qsu_x_cal(qsu,qsu_x,nx)
    # Initial c
        if j_sus_ini==2:
            c,wfcb=initial.c_init(c,qsu_x,wf,wfcb,alpha,nx)
            gcx=initial.gcx_init(c,gcx,nx,dx)

    # Seting for Plot
    y_h0=np.zeros([nx+1]); y_hc=np.zeros([nx+1])
    for i in np.arange(0,nx+1):
        y_h0[i]=h0_up[i]; y_hc[i]=eta_up[i]+hc_up[i]
    os.system("del /Q .\\png\\*.png")

    time=0.; err=0.; icount=0; nfile=0; l=0
    ysize=1.6*xsize

    ################ Main #####################

    while time<= etime:
        if icount%it_out==0 and time>o_start:
            print('time=',np.round(time,3),l)
            nfile=nfile+1

    # 計算結果のプロット

            fig=plt.figure(figsize = (xsize, ysize))

    # Upper Panel Left:Elevation Right: Width
            ax1= fig.add_subplot(4,1,1)
            im1= ax1.set_title("1D Open Channel Flow with Bed Deformation",fontsize=50)
            im1=ax1.set_xlim(xmin,xmax)
            im1=ax1.set_ylim(ymin,ymax)
            im1=ax1.set_title("Longitudinal Proile",size='30')
            im1=ax1.set_xlabel("x(m)",size='25')
            im1=ax1.set_ylabel("Elevation(m)",size='25')
            for tick in ax1.get_xticklabels():
                tick.set(fontsize=25)
            for tick in ax1.get_yticklabels():
                tick.set(fontsize=25)
            ax1.text(xmin,ymin,"Time="+str(np.round(time,3))+"sec",size=35)
        
            ax1r=ax1.twinx() #右軸    
            im1r=ax1r.set_ylim(bmin,bmax)
            im1r=ax1r.set_ylabel("Width(m)",fontsize=30)
            for tick in ax1r.get_yticklabels():
                tick.set(fontsize=25)

            im1= ax1.plot(x_cell[1:nx],eta[1:nx],'brown',label='Bed',linewidth=5)
            im1= ax1.plot(x_cell[1:nx],eta0[1:nx],linestyle="dashed",color='magenta')
            ax1.text(x_cell[1],eta[1],"Bed",size=30,color='brown')
            im1= ax1.plot(x_cell[1:nx],h[1:nx],'blue',label='WSE',linewidth=5)
            ax1.text(x_cell[1],h[1],"Water Surface",size=30,color='blue')
            im1r=ax1r.plot(x[0:nx],b_up[0:nx],'green',label='Width',linewidth=5)
            ax1r.text(x[nx]*.8,b_up[nx],"Channel Width",size=30,color='green')
            
    # Mid Pannel: Velocity Right: Tausta
            ax2=fig.add_subplot(4,1,2)
            im2=ax2.set_xlim(xmin,xmax)
            im2=ax2.set_xlabel("x(m)",size='25')
            im2=ax2.set_ylim(umin, umax)
            im2=ax2.set_ylabel("Velocity(m/s)",fontsize=30)
            for tick in ax2.get_xticklabels():
                tick.set(fontsize=25)
            for tick in ax2.get_yticklabels():
                tick.set(fontsize=25)
            im2= ax2.plot(x[0:nx],u[0:nx],'red',label='Velocity',linewidth=5)
            ax2.text(x[0],u[0],"Velocity",size=30,color='red')

            ax2r=ax2.twinx() # Right Hand Vertical Axis
            im2r=ax2r.set_ylim(tausta_min,tausta_max)
            im2r=ax2r.set_ylabel("Tausta",fontsize=30)
            for tick in ax2r.get_yticklabels():
                tick.set(fontsize=25)
            im2r=ax2r.plot(x[0:nx],tausta[0:nx],'blue',label='Tausta',linewidth=5)
            ax2r.text(x[nx]*.8,tausta[nx],"Tausta",size=30,color='blue')

            ax2.text(xmin,umin,"Time="+str(np.round(time,3))+"sec",size=35)

    # Next to Bottom Panel: Left:Discharge Right:Froude Number
            ax3= fig.add_subplot(4,1,3)
            im3=ax3.set_xlim(xmin,xmax)
            im3=ax3.set_xlabel("x(m)",size='25')
            im3=ax3.set_ylim(qmin, qmax)
            im3=ax3.set_ylabel("Discharge(m3/s)",fontsize=30)
            for tick in ax3.get_xticklabels():
                tick.set(fontsize=25)
            for tick in ax3.get_yticklabels():
                tick.set(fontsize=25)
            im3= ax3.plot(x[0:nx],qu[0:nx],'green',label='Dicharge',linewidth=5)
            ax3.text(x[0],qu[0],"Discharge",size=30,color='green')

            ax3r=ax3.twinx() # Right Hand Vertical Axis
            im3r=ax3r.set_ylim(frmin,frmax)
            im3r=ax3r.set_ylabel("Froude Number",fontsize=30)
            for tick in ax3r.get_yticklabels():
                tick.set(fontsize=25)
            im3r=ax3r.plot(x[0:nx],fr[0:nx],'red',label='Fr',linewidth=5)
            ax3r.text(x[nx]*.8,fr[nx],"Fr",size=30,color='red')

            ax3.text(xmin,qmin,"Time="+str(np.round(time,3))+"sec",size=35)

    # Bottom Panel: Sediment Discharfe
            ax4= fig.add_subplot(4,1,4)
            im4=ax4.set_xlim(xmin,xmax)
            im4=ax4.set_xlabel("x(m)",size='25')
            im4=ax4.set_ylim(qbmin, qbmax)
            im4=ax4.set_yscale("log")
            im4=ax4.set_ylabel("Sediment Discharge(m3/s)",fontsize=30)
            for tick in ax4.get_xticklabels():
                tick.set(fontsize=25)
            for tick in ax4.get_yticklabels():
                tick.set(fontsize=25)
            qbt=qb*b_up
            for i in np.arange(0,nx):
                if qbt[i]<=qbmin:
                    qbt[i]=qbmin

            im4= ax4.plot(x[0:nx],qbt[0:nx],'blue',label='Sediment Dicharge',linewidth=5)
            ax4.text(x[0],qbt[0],"Sediment Discharge",size=30,color='blue')

            ax4.text(xmin,qbmin,"Time="+str(np.round(time,3))+"sec",size=35)

            fname="./png/" + 'f%04d' % nfile + '.png'
            im=plt.savefig(fname)
            plt.clf()
            plt.close()
    #        plt.show();exit()

    # Non-Advection Phase
        l=0
        while l<lmax:
            hs_up=boundary.hs_up_cal(hs,hs_up,nx,hs_upstm,hs_dwstm,j_upstm,j_dwstm)
            cfx,ie=cfxx.cfx_cal(cfx,nx,un,hs_up,ie,g,snm)
            un=rhs.un_cal(un,u,nx,dx,cfx,hn,g,dt)
            un=boundary.u_bound(un,hs_up,qp,b_up,nx,j_upstm,j_dwstm,u_upstm,u_dwstm)
            qu=rhs.qu_cal(qu,un,hs_up,b_up,nx)
            hn,hs,err=hcal.hh(hn,h,hs,eta,qu,b,alh,hmin,dx,nx,dt,err)
            hn,hs=boundary.h_bound(hn,hs,eta,h_upstm,h_dwstm,hs_dwstm,nx,j_upstm,j_dwstm)

            if err<errmax:
                break
            l=l+1



    #Differentials in Non Advection Phase
        gux=newgrd.ng_u(gux,u,un,nx,dx)
        gux=boundary.gbound_u(gux,nx)

    # Advection Phase
        fn=np.zeros([nx2]);gxn=np.zeros([nx2])
    #    fn,gxn=mkzero.z0(fn,gxn,nx)
        fn,gxn=cip1d.u_cal1(un,gux,u,fn,gxn,nx,dx,dt)
        un,gux=cip1d.u_cal2(fn,gxn,u,un,gux,nx,dx,dt)
        un=boundary.u_bound(un,hs_up,qp,b_up,nx,j_upstm,j_dwstm,u_upstm,u_dwstm)
        gux=boundary.gbound_u(gux,nx)


    # Update u and h
        h=copy.copy(hn); u=copy.copy(un)

    # usta and tausta
        usta,tausta,fr=shear.usts(u,usta,tausta,fr,ie,hs_up,g,sd,nx)
        usta,tausta,fr=boundary.shear_bound(usta,tausta,fr)
        usta_x=boundary.usta_xcal(usta,usta_x,nx)

    # Bedload Sediment Transport
        if j_sediment==1 or j_sediment==3:
            qb=qbz.qbcal(qb,tausta,tsc,sgd3_s,nx,eta,dx,gamma_1)
            qb=boundary.qb_bound(qb,alpha_up_sed)

    # Supended Sediment
        if j_sediment>=2:
            alpha,beta=phisval.alpha_beta(alpha,beta,usta_x,wf,nx)
            qsu=qsucal.itakura(qsu,tausta,sgd_s,wf,nx)
            qsu=boundary.qsu_bound(qsu)
            qsu_x=boundary.qsu_x_cal(qsu,qsu_x,nx)
    # Sediment Source, Diffusion and Advection
            cn,wfcb=source.cn1(c,cn,qsu_x,hs,alpha,wf,wfcb,nx,dt)
            cn,wfcb=boundary.cbound(cn,wfcb,nx)
            epsilon=eddyvis.epcal(epsilon,usta,hs_up,kappa,nx)
            cn,dcdx=diffusion.cdiff(cn,epsilon,dcdx,nx,dx,dt)
            gcx=newgrd.gcxcal(c,cn,gcx,nx,dx)
            fn=np.zeros([nx2]);gxn=np.zeros([nx2])
    #        fn,gxn=mkzero.z0(fn,gxn,nx)
            fn,gxn=cip1d.c_cal1(cn,gcx,u,fn,gxn,nx,dx,dt)
            cn,gcx=cip1d.c_cal2(fn,gxn,c,cn,gcx,nx,dx,dt)
            cn,wfcb=boundary.cbound(cn,wfcb,nx)
            gcx=boundary.gbound_c(gcx,nx)

    # Bed Deformation
        if j_sediment>=1 and time>=b_start:
            eta,deta,hs=qbz.etacal_bl(eta,deta,qb,qsu_x,h,hs,b_up,b,nx,dx,wf,wfcb,rlam,cn,alpha,j_sediment)
            eta,hs,h=boundary.eta_bound(eta,h,hs,nx)
            eta_up=boundary.eta_up_cal(eta,eta_up,nx)
    # Update C
        if j_sediment>=2:
            c=copy.copy(cn)

    #Time Step Update
        time=time+dt
        icount=icount+1

    os.system("del /Q *.mp4")
    os.system("del /Q *.gif")

    subprocess.call('ffmpeg -framerate 30 -i png/f%4d.png -r 60 -an -vcodec libx264 -pix_fmt yuv420p egexp.mp4',  shell=True)
    os.system('ffmpeg -i egexp.mp4 egexp.gif -loop 0')

    folder_name="ans_save\\"+case_name
    target=["ini_longitudinal_prof.png","egexp.gif","egexp.mp4"]
    for tg in target:
        cmd_name="copy /Y "+tg+" "+folder_name
        os.system(cmd_name)

    cmd="mkdir "+folder_name+"\\png"
    os.system(cmd)
    cmd="del /Q "+folder_name+"\\png\\*.png"
    os.system(cmd)
    cmd="copy /Y png\\*.png "+folder_name+"\\png"
    os.system(cmd)

