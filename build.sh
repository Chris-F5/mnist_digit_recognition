#!/bin/bash

CFLAGS=-g
LDFLAGS=-lm

gcc $CFLAGS -c main.c data.c
gcc $CFLAGS $LDFLAGS -omain main.o data.o
