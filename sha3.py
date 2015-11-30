# -*- coding: utf-8 -*-
from __future__ import division
import sys
import pyopencl as cl
import numpy as np
import pylab
import pdb
import time

def pad10star1(M, n):
    """Pad M with the pad10*1 padding rule to reach a length multiple of r bits

    M: message pair (length in bits, string of hex characters ('9AFC...')
    n: length in bits (must be a multiple of 8)
    Example: pad10star1([60, 'BA594E0FB9EBBD30'],8) returns 'BA594E0FB9EBBD93'
    """

    [my_string_length, my_string]=M

    # Check the parameter n
    if n%8!=0:
        raise KeccakError.KeccakError("n must be a multiple of 8")

    # Check the length of the provided string
    if len(my_string)%2!=0:
        #Pad with one '0' to reach correct length (don't know test
        #vectors coding)
        my_string=my_string+'0'
    if my_string_length>(len(my_string)//2*8):
        raise KeccakError.KeccakError("the string is too short to contain the number of bits announced")

    nr_bytes_filled=my_string_length//8
    nbr_bits_filled=my_string_length%8
    l = my_string_length % n
    if ((n-8) <= l <= (n-2)):
        if (nbr_bits_filled == 0):
            my_byte = 0
        else:
            my_byte=int(my_string[nr_bytes_filled*2:nr_bytes_filled*2+2],16)
        my_byte=(my_byte>>(8-nbr_bits_filled))
        my_byte=my_byte+2**(nbr_bits_filled)+2**7
        my_byte="%02X" % my_byte
        my_string=my_string[0:nr_bytes_filled*2]+my_byte
    else:
        if (nbr_bits_filled == 0):
            my_byte = 0
        else:
            my_byte=int(my_string[nr_bytes_filled*2:nr_bytes_filled*2+2],16)
        my_byte=(my_byte>>(8-nbr_bits_filled))
        my_byte=my_byte+2**(nbr_bits_filled)
        my_byte="%02X" % my_byte
        my_string=my_string[0:nr_bytes_filled*2]+my_byte
        while((8*len(my_string)//2)%n < (n-8)):
            my_string=my_string+'00'
        my_string = my_string+'80'

    return my_string

def KeccakF(to_hash, iterations, curr_iter, program, context, queue):



    WORDLENGTH = 64
    inputnum = int(to_hash.shape[0]/5)
    #Set up Round constants
    RC=[0x0000000000000001,
        0x0000000000008082,
        0x800000000000808A,
        0x8000000080008000,
        0x000000000000808B,
        0x0000000080000001,
        0x8000000080008081,
        0x8000000000008009,
        0x000000000000008A,
        0x0000000000000088,
        0x0000000080008009,
        0x000000008000000A,
        0x000000008000808B,
        0x800000000000008B,
        0x8000000000008089,
        0x8000000000008003,
        0x8000000000008002,
        0x8000000000000080,
        0x000000000000800A,
        0x800000008000000A,
        0x8000000080008081,
        0x8000000000008080,
        0x0000000080000001,
        0x8000000080008008]

    round_constants = np.array([np.uint64(x) for x in RC])
    round_constants_gpu = cl.Buffer(context, cl.mem_flags.READ_ONLY, 8 * len(round_constants))   #Why * 8???
    cl.enqueue_copy(queue, round_constants_gpu, round_constants, is_blocking=False)

    #Set up rotation offsets
    rotation_offsets = np.array([0, 36, 3, 41, 18, \
                                1 ,44 ,10 ,45 ,2,   \
                                62, 6 ,43 ,15 ,61,  \
                                28, 55, 25, 21, 56, \
                                27, 20, 39, 8 ,14])

    rotation_offsets = np.array([np.uint64(x) for x in rotation_offsets])
    rotation_gpu_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY, 8 * len(rotation_offsets))
    cl.enqueue_copy(queue, rotation_gpu_buffer, rotation_offsets, is_blocking=False)




    stuff_to_hash = cl.Buffer(context, cl.mem_flags.READ_ONLY, to_hash.size * 8)
    cl.enqueue_copy(queue, stuff_to_hash, to_hash, is_blocking=False)#is_block=True means wait for completion

    #Buffer for GPU to write final hash
    gpu_final_hash = cl.Buffer(context, cl.mem_flags.READ_WRITE, to_hash.size * 8)
    
    #control the number of iterations of each hash in Keccak
    gpu_iterations = cl.Buffer(context, cl.mem_flags.READ_ONLY, len(iterations)*8)
    cl.enqueue_copy(queue, gpu_iterations, np.array(iterations), is_blocking=False)#is_block=True means wait for completion

    gpu_curr_iter = cl.Buffer(context, cl.mem_flags.READ_ONLY, 8)
    cl.enqueue_copy(queue, gpu_curr_iter, np.array(curr_iter), is_blocking=False)#is_block=True means wait for completion


    #Create 5x5 workgroup, local buffer
    local_size, global_size = (5, 5) , (5,5*inputnum)
    local_buf_w,local_buf_h = np.uint64(5),np.uint64(5)
    A = cl.LocalMemory(8*25)
    B = cl.LocalMemory(8*25)
    C = cl.LocalMemory(8*25)
    D = cl.LocalMemory(8*25)

    #Hash input
    final_hash = np.zeros((5*inputnum,5))
    final_hash = np.array([np.uint64(x) for x in final_hash])    
    hash_event = program.sha_3_hash(queue, global_size, local_size,
                              stuff_to_hash, gpu_final_hash,rotation_gpu_buffer,round_constants_gpu, gpu_iterations, gpu_curr_iter,
                              B,A, C, D, local_buf_w,local_buf_h)

    
    
    cl.enqueue_copy(queue, final_hash, gpu_final_hash, is_blocking=True)

    return final_hash


def Keccak(inputlist, n,r,c, program, context, queue):



    inputnum = len(inputlist)
    input_str = inputlist[0]

    #P is a storage for the padded inputs
    P = []

    #Z is a storage for the output hashes
    Z = []

    iterations = []

    #start = time.time()
    ### Padding Phase
    for i in range(inputnum):
        tmpstr = pad10star1([len(inputlist[i])*4, inputlist[i]],r) 
        P.append(tmpstr)
        Z.append("")
        iterations.append((len(tmpstr)*8//2)//r)

    #print "Time to run padding: " + str(time.time() - start)

    # Initialisation of state
    S = np.zeros((5*inputnum,5))


    #Testing
    S = np.array([np.uint64(x) for x in S])

    #Initialize workgroup sizes for the gpu
    local_size, global_size = (5, 5) , (5,5*inputnum)

    #THIS PART HAS TO CHANGE
    for i in range(max(iterations)): 

        host_string = ""
        Pi = np.zeros((5*inputnum,5))
        Pi = np.array([np.uint64(x) for x in Pi])

        for j in range(inputnum):

            if (iterations[j] > i):
                #Absorbing Phase
                host_string = host_string + str(P[j][i*(2*r//8):(i+1)*(2*r//8)]+'00'*(c//8))
            else:
                #Dummy variables. Won't be used.
                host_string = host_string + "0"*400

        gpu_string = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_string)
        gpu_table = cl.Buffer(context, cl.mem_flags.READ_WRITE, 25*8 * inputnum)
        #gpu_iterations = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=iterations)
        part_of_string = cl.LocalMemory(1*16)
        program.convert_str_to_table(queue,global_size,local_size, gpu_string, gpu_table, part_of_string, np.uint64(5),np.uint64(5),np.uint64(64))
        cl.enqueue_copy(queue, Pi, gpu_table, is_blocking=True)




        for x in range(5*inputnum):
            for y in range(5):
                #print 'type S:',type(S[x][y]),'type P:',type(Pi[x][y])

                if (iterations[int(x/5)] > i):
                    S[x][y] = S[x][y]^Pi[x][y]



        S = np.array([np.uint64(x) for x in S])
        #start = time.time()
        S = KeccakF(S, iterations, i,  program, context, queue)
        #print "Time to run KeccakF: " + str(time.time() - start)
        #print S

    #Squeezing phase

    outputstring = np.chararray(400 * inputnum)
    gpu_table = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=S)
    gpu_string = cl.Buffer(context, cl.mem_flags.READ_WRITE, 144*8 * inputnum)
    program.convert_table_to_str(queue,global_size,local_size,gpu_table, gpu_string,np.uint64(5),np.uint64(5),np.uint64(64))
    cl.enqueue_copy(queue, outputstring, gpu_string, is_blocking=True)

    string = ''.join(outputstring)

    for x in range(inputnum):
        Z[x] = Z[x] + string[400 * x: 400 * x + r*2//8]


    for x in range(inputnum):
        Z[x] = Z[x][0:2*n//8]

    #output the pre-set number of bits
    return Z

if __name__ == '__main__':


    # List our platforms
    platforms = cl.get_platforms()

    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices[:2])
    #print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    #print 'The queue is using the device:', queue.device.name

    
    program = cl.Program(context, open('sha3.cl').read()).build(options='')


    #PARAMETERS for SHA 512
    r = 576
    c = 1024
    n = 512

    inputlist = []
    inputlist.append("")
    inputlist.append("abcd")
    inputlist.append("abcd")
    inputlist.append("abcd")
    inputlist.append("a" * 1000)

    start = time.time()
    result = Keccak(inputlist, n, r,c, program, context, queue)
    print  "Hashing Result is"
    print result
    print "Time taken is: " + str(time.time() - start)