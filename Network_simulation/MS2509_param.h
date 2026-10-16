/* MS2509 _param.h - Parameters, Macros, Enums */
#ifndef PARAM_H
#define PARAM_H

#define MIN_HOSTS   3
#define MAX_HOSTS   26
#define BUF         256     /* port buffer size d[2][256]   */
#define NET_ADDR    7       /* simulated network address     */
#define MAX_LOG     50

/* SMAC: single letter A=host0, B=host1, C=host2 ...        */
#define SMAC(i)  ('A' + (i))

/* prob(P): returns 1 roughly P% of the time               */
#define PROB(P)  ((rand() % 100) < (P))

typedef enum { SUCCESS, COLLISION, DROPPED } TxResult;

#endif