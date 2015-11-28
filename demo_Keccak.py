# -*- coding: utf-8 -*-
#! /usr/bin/pythonw
# The Keccak sponge function, designed by Guido Bertoni, Joan Daemen,
# Michaël Peeters and Gilles Van Assche. For more information, feedback or
# questions, please refer to our website: http://keccak.noekeon.org/
# 
# Implementation by Renaud Bauvin,
# hereby denoted as "the implementer".
# 
# To the extent possible under law, the implementer has waived all copyright
# and related or neighboring rights to the source code in this file.
# http://creativecommons.org/publicdomain/zero/1.0/

import time

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

import Keccak

myKeccak=Keccak.Keccak(1600)

start = time.time()
print myKeccak.Keccak([0,""],r = 576, c=1024,n=512, verbose=False)
print "total time:" + str(time.time() - start)

start = time.time()
print myKeccak.Keccak([16,"abcd"],r = 576, c=1024,n=512, verbose=False)
print "total time:" + str(time.time() - start)

start = time.time()
print myKeccak.Keccak([4000,"a"*1000],r = 576, c=1024,n=512, verbose=False)
print "total time:" + str(time.time() - start)