/* MS2509_fun.h - Structs and Function Declarations */
#ifndef FUN_H
#define FUN_H

#include "MS2509_param.h"

/*
 * PORT: each host has two ports (one to TestP, one to LAN).
 * d[0][*] = out-buffer   d[1][*] = in-buffer
 * d[*][0] = byte count (0 means empty)
 */
typedef struct {
    unsigned char d[2][BUF];
} Port;

/* One host: two ports + stats */
typedef struct {
    int   id;           /* 0,1,2,...        */
    char  smac;         /* A,B,C,...        */
    Port  tp_port;      /* TestP <-> LLL    */
    Port  lan_port;     /* LLL  <-> LAN     */
    int   sent;
    int   received;
    int   busy;         /* processing ticks left */
} Host;

/* Simulation state */
typedef struct {
    Host  hosts[MAX_HOSTS];
    int   n;            /* number of hosts  */
    int   iter;         /* iteration counter*/
    int   total_sent;
    int   total_collisions;
} Sim;

/* ---------- function declarations ---------- */
void  init_sim   (Sim *s, int n);
void  scheduler  (Sim *s);          /* one iteration         */
void  strip_testp(Sim *s, int i);   /* TestP strip for host i*/
void  strip_lll  (Sim *s, int i);   /* LLL strip for host i  */
void  strip_lan  (Sim *s);          /* LAN strip             */
void  connectors (Sim *s);          /* move data port→port   */
void  show_hosts (Sim *s);
void  show_stats (Sim *s);

#endif