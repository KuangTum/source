/* OverlapOnlyDump.c : dump a minimal .scfout containing only OLP (S) and OLP_r (x,y,z)
 * for OpenMX 3.9.9, to be consumed by HopTB's createmodelopenmx_overlaponly().
 *
 * Build: add OverlapOnlyDump.o to Makefile objects, and compile with -DOLPR_DUMP
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_common.h"
#include "Inputtools.h"
#include "mpi.h"
/* 用来计算每原子轨道数 TNO[i] = Spe_Total_CNO[ WhatSpecies[i] ] */
extern int *WhatSpecies;     /* 1..atomnum → 物种索引 */
extern int *Spe_Total_CNO;   /* 1..SpeciesNum → 该物种 AO 数 */

/* ====== 可能因版本差异需用 grep 核对的数组名 ======
 *  1) Overlap：通常是 double**** OLP  （或 dcomplex**** OLP；此处按实数写）
 *  2) 位置重叠（<phi|r|phi>）：
 *     常见为 OLPpo[3][order][ct_AN][h_AN][i][j] 或 RCN_x/y/z[ct_AN][h_AN][i][j]
 *     用 grep -n "Position matrix elements" 或 "OLPpo" / "RCN_" 确认。
 */

/* ------- 写工具：按 HopTB 解析器要求的“4 元组打包”写 3D 向量 ------- */
static void write_3d_vecs_double(FILE *fp, int num, double **vec) {
  /* HopTB 读法是 reshape(multiread(T,4*num),4,num)[2:4,:]
     也就是说每个向量写 4 个 double，第一个值会被丢弃。这里写 0 + (x,y,z)。 */
  for (int k=0; k<num; ++k) {
    double pack[4] = {0.0, vec[k][1], vec[k][2], vec[k][3]};
    fwrite(pack, sizeof(double), 4, fp);
  }
}

static void write_3d_vecs_int(FILE *fp, int num, int **vec) {
  for (int k=0; k<num; ++k) {
    int pack[4] = {0, vec[k][1], vec[k][2], vec[k][3]};
    fwrite(pack, sizeof(int), 4, fp);
  }
}

/* ------- 把一个邻接块矩阵按 Julia 的列主序写出 -------
 * HopTB 在读每个块时先分配 Matrix{Float64}(rows, cols)
 * rows = Total_NumOrbs[natn[i][j]], cols = Total_NumOrbs[i]
 * 然后直接 read!(f, t)，因此我们必须按“列主序”写出。
 * 另外注意 OpenMX 内部 OLP[ct][h][i][j] 的 i/j 是 (中心原子轨道, 邻居轨道)，
 * 而 HopTB 的 t[jj,ii] 期望 (行=邻居轨道, 列=中心轨道)，等价于做一次转置：
 *   t(jj,ii) = OLP[ct][h][ii][jj]
 */
/* 写 4* 指针的一个块（用于 r 矩阵） */
static void dump_block_matrix4(FILE *fp,
                               int tno_center, int tno_neigh,
                               double ****M, int ct_AN, int h_AN)
{
  for (int ii=0; ii<tno_center; ++ii) {       /* 列：中心原子轨道 */
    for (int jj=0; jj<tno_neigh;  ++jj) {     /* 行：邻居原子轨道 */
      double v = M[ct_AN][h_AN][ii][jj];      /* 如需转置，这里改成 [jj][ii] */
      fwrite(&v, sizeof(double), 1, fp);
    }
  }
}

/* 写 5* 指针的一个块（用于 S，带自旋通道） */
static void dump_block_matrix5(FILE *fp,
                               int tno_center, int tno_neigh,
                               double *****M, int spin, int ct_AN, int h_AN)
{
  for (int ii=0; ii<tno_center; ++ii) {       /* 列：中心原子轨道 */
    for (int jj=0; jj<tno_neigh;  ++jj) {     /* 行：邻居原子轨道 */
      double v = M[spin][ct_AN][h_AN][ii][jj];
      fwrite(&v, sizeof(double), 1, fp);
    }
  }
}

/* 同上，但用于 3 个方向的 <r>；把指针数组传进来 */
static void dump_block_matrix_r(FILE *fp,
                                int tno_center, int tno_neigh,
                                double ****RX, double ****RY, double ****RZ,
                                int dir, int ct_AN, int h_AN)
{
  double ****ARR = (dir==0?RX:(dir==1?RY:RZ));
  for (int ii=0; ii<tno_center; ++ii) {
    for (int jj=0; jj<tno_neigh;  ++jj) {
      double v = ARR[ct_AN][h_AN][ii][jj];
      fwrite(&v, sizeof(double), 1, fp);
    }
  }
}

/* 计算每原子的轨道起始偏移（用于 set* 时映射；这里只写文件不需要，但保留示例） */
static void calc_numorb_base(int atomnum, int *Total_NumOrbs, int *base)
{
  base[1] = 0;
  for (int i=2; i<=atomnum; ++i) base[i] = base[i-1] + Total_NumOrbs[i-1];
}

/* ====== 导出最小 .scfout：头/平移/邻接/tv-rtv-Gxyz/OLP/OLP_r ====== */
void Dump_OverlapOnly_SCFOUT(const char *fname,
                             /* 输入：OLP 与位置重叠的三个方向 */
                             double *****OLP_arr,
                             double ****RX_arr,
                             double ****RY_arr,
                             double ****RZ_arr)
{
  int myid=0;
#ifdef MPI
  MPI_Comm_rank(mpi_comm_level1, &myid);
#endif
  if (myid!=0) return;  /* 只由 rank 0 写一个文件 */

  FILE *fp = fopen(fname, "wb");
  if (!fp) {
    if (myid==0) printf("Cannot open dump file %s\n", fname);
    return;
  }

  /* ---------- 1) header（7 个 Int32） ---------- */
  /* HopTB 读入时要求 (SpinP_switch >> 2) == 3，随后 &0x03 得到自旋类型。
     因此我们把高两位写成 3（OpenMX 标签），低两位保留当前 SpinP_switch。 */
  int SpinP_masked = SpinP_switch & 0x03;       /* 0:非自旋, 1:共线, 3:NC */
  int SpinP_packed = (3<<2) | SpinP_masked;     /* 高位标签 */
  int order_max    = 1;                         /* 只写 1 阶 r 矩阵 */

int Catomnum_loc = atomnum;  /* 周期体：central 区域=整个胞 */
int Latomnum_loc = 0;        /* 无左右电极 */
int Ratomnum_loc = 0;

int hdr[7];
hdr[0] = atomnum;
hdr[1] = SpinP_packed;
hdr[2] = Catomnum_loc;
hdr[3] = Latomnum_loc;
hdr[4] = Ratomnum_loc;
hdr[5] = TCpyCell;
hdr[6] = order_max;
fwrite(hdr, sizeof(int), 7, fp);


  /* ---------- 2) 平移表 atv / atv_ijk ---------- */
  /* C 里是 0..TCpyCell 共 TCpyCell+1 项 */
  write_3d_vecs_double(fp, TCpyCell+1, atv);
  write_3d_vecs_int   (fp, TCpyCell+1, atv_ijk);

  /* ---------- 3) Total_NumOrbs, FNAN, natn, ncn ---------- */
  int *TNO = (int*)malloc(sizeof(int)*(atomnum+1));
  for (int i=1; i<=atomnum; ++i) {
    int sp = WhatSpecies[i];
    TNO[i] = Spe_Total_CNO[sp];
  }
  /* 注意：scfout 里 FNAN 存“去 self 的个数”，HopTB 读出后会 +1；
           但 natn/ncn 的数组长度是 (FNAN+1)，包含 self(h=0)。 */
  /* 3.1 Total_NumOrbs (每原子轨道数) */
  {
    int *tmp = (int*)malloc(sizeof(int)*(atomnum));
    for (int i=1;i<=atomnum;++i) tmp[i-1] = TNO[i];
    fwrite(tmp, sizeof(int), atomnum, fp);
    free(tmp);
  }

  /* 3.2 FNAN（去 self） */
  {
    int *tmp = (int*)malloc(sizeof(int)*(atomnum));
    for (int i=1;i<=atomnum;++i) tmp[i-1] = FNAN[i]; /* 注意：OpenMX 内部 FNAN 是“不含 self 的个数”？ */
    /* 如果你源码里循环邻接是 0..FNAN[i]，说明内存中的 FNAN 是“含 self 的个数”；
       那么此处应写 FNAN[i]-1。用下面这一行替换上一行： */
    /* tmp[i-1] = FNAN[i] - 1; */
    fwrite(tmp, sizeof(int), atomnum, fp);
    free(tmp);
  }

  /* 3.3 natn：每原子写 (FNAN+1) 个条目，包含 self 在第 0 项 */
  for (int i=1;i<=atomnum;++i) {
    int len = FNAN[i] + 1;              /* 若你的 FNAN 已经“不含 self”，改成 FNAN[i]+1 */
    int *buf = (int*)malloc(sizeof(int)*len);
    for (int h=0; h<len; ++h) buf[h] = natn[i][h]; /* self: natn[i][0] 应等于 i */
    fwrite(buf, sizeof(int), len, fp);
    free(buf);
  }

  /* 3.4 ncn：同上，注意文件里用 0-based，HopTB 读后 +1 */
  for (int i=1;i<=atomnum;++i) {
    int len = FNAN[i] + 1;
    int *buf = (int*)malloc(sizeof(int)*len);
    for (int h=0; h<len; ++h) buf[h] = ncn[i][h] -1;   /* 如果内存是 1-based 索引，需要 -1 */
    /* 若确认 ncn[i][h] 是 1..TCpyCell，则用： buf[h] = ncn[i][h] - 1; */
    fwrite(buf, sizeof(int), len, fp);
    free(buf);
  }

  /* ---------- 4) tv, rtv, Gxyz （均以 Bohr 写出） ---------- */
  double *tv_ptr[3]  = { tv[1],  tv[2],  tv[3] };
  double *rtv_ptr[3] = { rtv[1], rtv[2], rtv[3] };
  write_3d_vecs_double(fp, 3, tv_ptr);
  write_3d_vecs_double(fp, 3, rtv_ptr);
  /* Gxyz: 原子笛卡尔坐标（Bohr）。OpenMX 的 Gxyz 通常为 Gxyz[1..3][1..atomnum] */
  {
    /* 组装成 [atom][4] */
    double *buf_flat = malloc(sizeof(double)*atomnum*4);
    double **buf = malloc(sizeof(double*)*atomnum);
    for (int a=0; a<atomnum; ++a) buf[a] = buf_flat + 4*a;
    for (int a=1; a<=atomnum; ++a) {
      buf[a-1][0] = 0.0;
      buf[a-1][1] = Gxyz[1][a];
      buf[a-1][2] = Gxyz[2][a];
      buf[a-1][3] = Gxyz[3][a];
    }
    write_3d_vecs_double(fp, atomnum, buf);
    free(buf_flat);
  }

  /* ---------- 5) OLP：支持 0/1/NC 自旋 ---------- */
  for (int spin=0; spin<=SpinP_switch; ++spin) {
    for (int i=1;i<=atomnum;++i) {
      int len = FNAN[i] + 1;                    /* 含 self */
      for (int h=0; h<len; ++h) {
        int B = natn[i][h];
        int tno_center = TNO[i];
        int tno_neigh  = TNO[B];
        dump_block_matrix5(fp, tno_center, tno_neigh, OLP_arr, spin, i, h);
      }
    }
  }


  /* ---------- 6) OLP_r：3 个方向（order_max=1） ---------- */
  for (int dir=0; dir<3; ++dir) {
    /* 如果你的源码是 OLPpo[3][order][ct][h][i][j]，请先把 RX/RY/RZ 指向 OLPpo[0/1/2][0] */
    for (int i=1;i<=atomnum;++i) {
      int len = FNAN[i] + 1;
      for (int h=0; h<len; ++h) {
        int B = natn[i][h];
        int tno_center = TNO[i];
        int tno_neigh  = TNO[B];
        dump_block_matrix_r(fp, tno_center, tno_neigh, RX_arr, RY_arr, RZ_arr, dir, i, h);
      }
    }
  }
  free(TNO);
  fclose(fp);
  if (myid==0) {
    printf("[OverlapOnlyDump] wrote %s (S and r_x/y/z only)\n", fname);
  }
}
