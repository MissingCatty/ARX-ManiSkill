// CPU reference geometry. No network, robot control, or candidate fallback.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" int zg_detect(int64_t n, const double* poses, int64_t np,
    const double* points, uint8_t* bad, uint8_t* final_bad,
    int64_t* counts, double* widths, double* depths) {
  if (np == 0) return 1;
  std::vector<double> local(3*np);
  for (int64_t k=0; k<n; ++k) {
    const double* t=poses+16*k;
    // GraspNet +X=EEF+Z, +Y=EEF+Y, +Z=-EEF+X.
    // Native initial depth=40mm; its endpoint is the saved TCP43 contact.
    double dl=.044, dr=.044; int64_t il=-1, ir=-1;
    counts[k]=0;
    for (int64_t j=0; j<np; ++j) {
      double v[3]={points[3*j]-t[3],points[3*j+1]-t[7],points[3*j+2]-t[11]};
      double x=v[0]*t[2]+v[1]*t[6]+v[2]*t[10]+.04;
      double y=v[0]*t[1]+v[1]*t[5]+v[2]*t[9];
      double z=-(v[0]*t[0]+v[1]*t[4]+v[2]*t[8]);
      local[3*j]=x;local[3*j+1]=y;local[3*j+2]=z;
      bool inner=std::abs(z)<.01 && x>-.02 && x<.04 && y>=-.044 && y<=.044;
      counts[k]+=inner;
      double cy=inner?y:0.;
      if (il<0 || .044+cy<dl) {dl=.044+cy;il=j;}
      if (ir<0 || .044-cy<dr) {dr=.044-cy;ir=j;}
    }
    double d=std::min(dl,dr);
    widths[k]=.088+2*(std::clamp(d,.0025,.01)-d);
    depths[k]=std::max(0.,std::min(std::max(local[3*il],local[3*ir]),.04));
    bad[k]=counts[k]<10;final_bad[k]=bad[k];
    for (int64_t j=0; j<np; ++j) {
      const double x=local[3*j], y=local[3*j+1], z=local[3*j+2];
      if (std::abs(z)>=.01) continue;
      const bool base=x<-.02 && x>-.03 && y>-.054 && y<.054;
      const bool finger=x>-.02 && x<depths[k] &&
          ((y>-.054 && y<-.044)||(y>.044 && y<.054));
      bad[k]|=base||finger;
      const double h=widths[k]/2;
      const bool base2=x<-.02 && x>-.03 && y>-h-.01 && y<h+.01;
      const bool finger2=x>-.02 && x<depths[k] &&
          ((y>-h-.01 && y<-h)||(y>h && y<h+.01));
      final_bad[k]|=base2||finger2;
    }
    final_bad[k]|=widths[k]>.088+1e-12 || widths[k]<=0;
  }
  return 0;
}

extern "C" int arx_variable_collision(int64_t n,const double* poses,
    const double* widths,const double* points,const int64_t* offsets,
    const int64_t* ids,int64_t parts,const double* bounds,const double* planes,
    const int64_t* poff,const double* direction,double margin,uint8_t* bad) {
  std::vector<double> local;
  for (int64_t k=0;k<n;++k) {
    const double* t=poses+16*k;auto count=offsets[k+1]-offsets[k];
    local.resize(3*count);
    for (int64_t j=0;j<count;++j) {
      const double* p=points+3*ids[offsets[k]+j];
      double x=p[0]-t[3],y=p[1]-t[7],z=p[2]-t[11];
      for (int a=0;a<3;++a) local[3*j+a]=(x*t[a]+y*t[4+a])+z*t[8+a];
    }
    bad[k]=0;
    for (double advance : {-.08,-.04,0.}) {
      for (int64_t part=0;part<parts && !bad[k];++part) {
        if(poff[part]==poff[part+1]) continue;
        const double* b=bounds+6*part;
        double shift=(widths[k]/2-.044)*direction[part];
        for (int64_t j=0;j<count;++j) {
          double x=local[3*j], y=local[3*j+1]-shift,z=local[3*j+2]-advance;
          if(x<b[0]+margin||x>b[3]-margin||y<b[1]+margin||y>b[4]-margin||z<b[2]+margin||z>b[5]-margin)continue;
          bool inside=true;
          for(auto f=poff[part];f<poff[part+1];++f) {
            const double* p=planes+4*f;
            if(((x*p[0]+y*p[1])+z*p[2])+p[3]>-margin){inside=false;break;}
          }
          if(inside){bad[k]=1;break;}
        }
      }
      if(bad[k])break;
    }
  }
  return 0;
}
