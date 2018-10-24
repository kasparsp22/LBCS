from math import *
from AIPS import AIPS, AIPSDisk
from AIPSTask import AIPSTask, AIPSList
from AIPSData import AIPSUVData, AIPSImage, AIPSCat
from Wizardry.AIPSData import AIPSUVData as WizAIPSUVData
from Wizardry.AIPSData import AIPSImage as WizAIPSImage
import re, sys, numpy as np, os, pyfits, matplotlib
from matplotlib import pyplot as plt; from pyfits import getdata
from closure_trans import *
from scipy.fftpack import *
from scipy.linalg import *
plt.rcParams['image.origin']='lower'
plt.rcParams['image.interpolation']='nearest'
INDE = 3140.892822265625    # value corresponding to aips INDE
AIPSUSER = 2
AIPS.userno = AIPSUSER

# convert a difmap modelfile to a uv dataset

def dif2cc(modfile,aipsno,indisk=1):
    f = open(modfile)
    for line in f:
        if '!' in line and 'Center' in line:
            l = line.split()
            rahr,ramin = float(l[3]), float(l[4])
            rasec = float(l[5].replace(',',''))
            decdeg, decmin = float(l[7]), float(l[8])
            decsec = float(l[9])
            ra = 15.*(rahr+ramin/60.+rasec/3600.)
            dec = decdeg+decmin/60.+decsec/3600.
            cosdec = np.cos(np.deg2rad(dec))
        if not '!' in line:
            l = line.split()
            flux,r,theta = float(l[0]),float(l[1]),np.deg2rad(float(l[2])+90.)
            new = np.array([-r*np.cos(theta)/1000.,r*np.sin(theta)/1000.,flux])
            try:
                cc = np.vstack((cc,new))
            except:
                cc = np.copy(new)
    f.close()
    mkcc (cc, aipsno, indisk=indisk, ra=ra, dec=dec)


# wrapper for uvcon
    
def douvcon (antfile,freq,dec,hastart,haend,tint,chwid,nchan,outname,\
             in2name='ZEROS',in2class='FITS',indisk=1,cmodel='CC',phserr=0.0,\
             amperr=0.0,cboth=1.0):
    uvcon = AIPSTask('uvcon')
    uvcon.infile = antfile if antfile[0] in './' else './'+antfile
    uvcon.in2name = in2name
    uvcon.in2class = in2class
    uvcon.in2disk = indisk
    uvcon.cmodel = cmodel
    uvcon.outname = outname
    uvcon.aparm[1:] = [freq,0.0,dec,hastart,haend,0.0,tint,chwid,nchan,0.0]
    uvcon.bparm[1:] = [0.0,0.0,-1.0*phserr,amperr,cboth,0.0,0.0,0.0,0.0,0.0]
    save_stdout = sys.stdout
    sys.stdout = open('uvcon.log','a')
    uvcon.inp()
    sys.stdout = save_stdout
    uvcon.go()
    ants = np.array([])
    f=open(antfile)
    flines = 0
    for line in f:
        flines += 1
        if '#' in line:
            thisant = line.split('#')[1].lstrip().rstrip()
            ants = np.append(ants,thisant.split()[0])
    f.close()
    AIPSImage(in2name,in2class,indisk,1).clrstat()
    if len(ants)==flines-3:   # if we have antennas for all lines except the first 3 header lines
        insertants (outname, ants)

# make a fits file from an array with a specified RA and DEC at the
# central pixel (as defined by AIPS, which is one above the actual centre)

def mkfits(a,cdelt=-1,outname='zeros.fits',ra=0.0,dec=60.0):
    hdu=pyfits.PrimaryHDU(a)
    hdu.header['CTYPE1']='RA---SIN'
    hdu.header['CTYPE2']='DEC--SIN'
    hdu.header['CTYPE3']='STOKES'
    hdu.header['CDELT1']=-cdelt
    hdu.header['CDELT2']=cdelt
    hdu.header['CDELT3']=1.0
    hdu.header['CRVAL1']=ra
    hdu.header['CRVAL2']=dec
    hdu.header['CRVAL3']=1.0
    hdu.header['CRPIX1']=0.5*float(len(a[0]))
    hdu.header['CRPIX2']=0.5*float(len(a[1]))+1.
    hdu.header['CRPIX3']=1.0
    hdu.header['EQUINOX']=2000.0
    hdu.writeto('zeros.fits',clobber=True)

# given a set of clean components, write an AIPS image with a CC file
#  cc in format (xoff/arcsec, yoff/arcsec, flux, [bmaj/arcs,bmin/arcs,bpa]
def mkcc(cc, aipsno, cdelt=-1, indisk=1, ra=0.0, dec=60.0):
    a=np.zeros((128,128))
    if cdelt==-1:
        cdelt = 1.2*(abs(np.ravel(cc[:,:2])).max()/50.0)/3600.0
        cdelt = 1.0/3600.0 if cdelt==0.0 else cdelt
    mkfits (a, cdelt=cdelt, dec=dec)
    for i in AIPSCat()[indisk]:
        if i['name']=='ZEROS' and i['klass']=='FITS':
            AIPSImage('ZEROS','FITS',indisk,i['seq']).clrstat()
            AIPSImage('ZEROS','FITS',indisk,i['seq']).zap()
    
    stdout = sys.stdout; sys.stdout = open('parseltongue.log','a')
    fitld = AIPSTask('fitld')
    fitld.datain = './zeros.fits'
    fitld.outname = 'ZEROS'
    fitld.outclass = 'FITS'
    fitld.outdisk = indisk
    fitld.inp()
    fitld.go()
    ccmod = AIPSTask ('ccmod')
    print '*',cc
    for i in cc:
        ccmod.inname = 'ZEROS'
        ccmod.inclass = 'FITS'
        ccmod.indisk = indisk
        xpix = float(64.0 - i[0]/(cdelt*3600.0))
        ypix = float(65.0 + i[1]/(cdelt*3600.0)) # nb aips convention
        ccmod.pixxy[1:] = [xpix,ypix]
        ccmod.flux = float(i[2])
        ccmod.opcode = 'POIN'
        ccmod.go()
    if cc.shape[1]==6:  # make all objects Gaussian
        ccgau = AIPSTask('ccgau')
        ccgau.inname = 'ZEROS'
        ccgau.inclass = 'FITS'
        ccgau.indisk = indisk
        ccgau.bmaj = 1.0
        ccgau.bmin = 1.0
        ccgau.factor = 1.0
        ccgau.go()
        for i in range(len(cc)):
            tabed = AIPSTask('tabed')
            tabed.inname = 'ZEROS'
            tabed.inclass = 'FITS'
            tabed.indisk = indisk
            tabed.inext = 'CC'
            tabed.optyp = 'REPL'
            tabed.bcount = tabed.ecount = i+1
            tabed.aparm[1] = 4
            tabed.keyvalue[1] = float(cc[i,3]/3600.0)
            tabed.go()
            tabed.aparm[1] = 5
            tabed.keyvalue[1] = float(cc[i,4]/3600.0)
            tabed.go()
            tabed.aparm[1] = 6
            tabed.keyvalue[1] = float(cc[i,5])
            tabed.go()

    sys.stdout.close(); sys.stdout = stdout

    
# gauss array: xoff yoff flux fwhm axrat pa
def gauss2uv (gauss=np.array([[0.,1.,1.,1.,1.,0.],[0.,0.,1.,1.,1.,0.]]),\
           userno=340,\
           anfile='./LOFAR4.UVCON',uvout='gauss2uv',indisk=1,\
           freq=0.15,dec=60.0,hastart=-6.0,haend=6.0,tint=15.0,\
           chwid=0.1,nchan=600):
    import njj
    zapexisting(uvout)
    ants = np.array([])
    f=open(anfile)
    for line in f:
        if '#' in line:
            thisant = line.split('#')[1].lstrip().rstrip()
            ants = np.append(ants,thisant.split()[0])
    maxdist = 3.0*(np.hypot(gauss[:,0],gauss[:,1]).max()+gauss[:,3].max())
    image = np.zeros((1024,1024))
    cdelt = maxdist/1024.
    for g in gauss:
        coord = np.array([511-g[0]/cdelt,512+g[1]/cdelt])
        image += njj.mkgauss([1024,1024],coord,g[2],g[3]/cdelt,g[4],g[5])
        print 'Making gaussian at pixel %f %f\n'%(coord[0],coord[1])
    mkfits(image,cdelt=cdelt/3600.,outname='zeros.fits',dec=dec)
    fitld = AIPSTask('fitld')
    fitld.datain = './zeros.fits'
    fitld.outname = 'ZEROS'
    fitld.outclass = 'FITS'
    fitld.outdisk = indisk
    fitld.go()
    douvcon ('LOFAR4.UVCON',freq,dec,hastart,haend,tint,chwid,nchan,\
             'gauss2uv',cmodel='IMAG')
    insertants ('gauss2uv',ants)
    
#def douvcon (antfile,freq,dec,hastart,haend,tint,chwid,nchan,outname,\
#             in2name='ZEROS',in2class='FITS',cmodel='CC'):

def zapexisting(uvout,indisk=1):
    for i in AIPSCat()[1]:
        if i['name']=='ZEROS' and i['klass']=='FITS':
            AIPSImage('ZEROS','FITS',indisk,i['seq']).clrstat()
            AIPSImage('ZEROS','FITS',indisk,i['seq']).zap()
        if i['name']==uvout and i['klass']=='UVCON':
            AIPSUVData(uvout,'UVCON',indisk,i['seq']).clrstat()
            AIPSUVData(uvout,'UVCON',indisk,i['seq']).zap()

def insertants (uvout,ants,indisk=1):
    for i in range(len(ants)):
        tabed = AIPSTask('tabed')
        tabed.inname = uvout
        tabed.inclass = 'UVCON'
        tabed.indisk = indisk
        tabed.bcount = i+1
        tabed.ecount = i+1
        tabed.aparm[1:] = [1.0,0.0,0.0,3.0,0.0]
        tabed.optype='REPL'
        tabed.inext='AN'
        tabed.keystrng=str(ants[i])
        tabed.go()
    
def cc2uv (cc=np.array([[0.0,1.0,1.0]]),userno=340,\
           anfile='./LOFAR4.UVCON',uvout='cc2uv',indisk=1,\
           freq=0.15,dec=60.0,hastart=-6.0,haend=6.0,tint=15.0,\
           chwid=0.1,nchan=600):
    AIPS.userno = userno
    ants = np.array([])
    f=open(anfile)
    for line in f:
        if '#' in line:
            thisant = line.split('#')[1].lstrip().rstrip()
            ants = np.append(ants,thisant.split()[0])
    zapexisting (uvout)
    mkcc (cc,AIPS.userno)
    douvcon (anfile,freq,dec,hastart,haend,tint,chwid,nchan,uvout)
    insertants (uvout,ants)

def doFITTP (userno = 2, indisk=1, inname='lbcs', inclass='UVCON', inseq=1, outname='lbcs.fits'):
    #AIPS.userno = userno
    fittp = AIPSTask('fittp')
    fittp.indisk = indisk
    fittp.inname = inname
    fittp.inclass = inclass
    fittp.inseq = inseq
    fittp.dataout = outname
    fittp.inp() 
    #fittp.go()
    


########

AIPS.userno=AIPSUSER
"""
#   components of source: xoffset/arcs, yoffset/arcs, flux/Jy, bmax/arcsec, bmin/arcsec, pa
cc=np.array([[0.,0.,1.,2.,1.,45.],[0.,1.,0.5,1.,2.,0.]])
mkcc(cc,AIPSUSER)
douvcon ('LOFAR.UVCON',freq=0.14,dec=60,hastart=-1.,haend=-0.95,tint=2.0,\
         chwid=0.04882812,nchan=64,outname='lbcs',\
         in2name='ZEROS',in2class='FITS',cmodel='CC',phserr=1.0,\
         amperr=0.0,cboth=0.0)
"""

doFITTP()
