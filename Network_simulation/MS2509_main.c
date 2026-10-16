/* MS2509_main.c - Main: scheduler loop */
#include <stdio.h>
#include "MS2509_fun.h"

int main(void) {
    Sim s;
    int n;
    char ch;

    printf("=== SC601 LAN Network Simulator ===\n");
    printf("Enter number of hosts (3-%d): ", MAX_HOSTS);
    scanf("%d", &n);
    if (n < MIN_HOSTS || n > MAX_HOSTS) {
        printf("Invalid. Must be %d-%d.\n", MIN_HOSTS, MAX_HOSTS);
        return 1;
    }

    init_sim(&s, n);
    show_hosts(&s);

    /* Scheduler loop: run until user presses Q */
    do {
        scheduler(&s);
        printf("------------------------------------------------------------------------\n");
        printf("ENTER Q to quit; ANY OTHER CHARACTER to continue: ");
        scanf(" %c", &ch);
        printf("------------------------------------------------------------------------\n");
    } while (ch != 'Q' && ch != 'q');

    printf("USER REQUESTED TERMINATION. Quitting.\n");
    printf("------------------------------------------------------------------------\n");
    show_stats(&s);
    return 0;
}