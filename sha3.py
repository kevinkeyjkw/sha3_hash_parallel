# -*- coding: utf-8 -*-
from __future__ import division
import sys
import pyopencl as cl
import numpy as np
import pylab
import pdb
import time



#Only pad for string messages not integers???
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

def KeccakF(to_hash, program):



    WORDLENGTH = 64
    inputnum = int(to_hash.shape[0]/5)
    print inputnum
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

    #print RC[1]
    #round_constants = np.array([int(x,16) for x in RC])
    #round_constants = np.array(RC)
    #round_constants2 = np.array([(x % (1<<WORDLENGTH)) for x in RC])
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
    
    #Create 5x5 workgroup, local buffer
    local_size, global_size = (5, 5) , (5,5*inputnum)
    local_buf_w,local_buf_h = np.uint64(5),np.uint64(5)


    #gpu_local_memory = cl.LocalMemory(to_hash.size * 8)
    #A = cl.LocalMemory(to_hash.size * 8)
    #B = cl.LocalMemory(to_hash.size * 8)
    A = cl.LocalMemory(8*25)
    B = cl.LocalMemory(8*25)
    C = cl.LocalMemory(8*25)
    D = cl.LocalMemory(8*25)

    #Hash input
    final_hash = np.zeros((5*inputnum,5))
    final_hash = np.array([np.uint64(x) for x in final_hash])    
    hash_event = program.sha_3_hash(queue, global_size, local_size,
                              stuff_to_hash, gpu_final_hash,rotation_gpu_buffer,round_constants_gpu,
                              B,A, C, D, local_buf_w,local_buf_h)

    
    
    cl.enqueue_copy(queue, final_hash, gpu_final_hash, is_blocking=True)

    #Profiling part
    seconds = (hash_event.profile.end - hash_event.profile.start) / 1e9
    print 'Total seconds to run rounds:',seconds


    #hex_output = [map(hex, l) for l in np.transpose(final_hash)]
    #hex_output = [map(hex, l) for l in final_hash]
    #print "output:"
    #for counter in range(inputnum):
    #    print "Input " + str(counter) + " Result:"
    #    for x in range(len(hex_output)):
    #        print hex_output[x][counter*5:counter*5+4]
    #print np.transpose(final_hash)

    #THIS PART HAS TO CHANGE
    return final_hash

def fromHexStringToLane(string):
    """Convert a string of bytes written in hexadecimal to a lane value"""

    #Check that the string has an even number of characters i.e. whole number of bytes
    if len(string)%2!=0:
        print "The provided string does not end with a full byte"
        exit(1)


    #Perform the modification
    temp=''
    nrBytes=len(string)//2
    for i in range(nrBytes):
        offset=(nrBytes-i-1)*2
        temp+=string[offset:offset+2]
    return np.uint64(int(temp,16))
    #return int(temp, 16)

def fromLaneToHexString(lane):
    """Convert a lane value to a string of bytes written in hexadecimal"""

    laneHexBE = (("%%0%dX" % (64//4)) % lane)
    #Perform the modification
    temp=''
    nrBytes=len(laneHexBE)//2
    for i in range(nrBytes):
        offset=(nrBytes-i-1)*2
        temp+=laneHexBE[offset:offset+2]
    return temp.upper()


### Conversion functions String <-> Table (and vice-versa)

def convertStrToTable(string):
    """Convert a string of bytes to its 5×5 matrix representation

    string: string of bytes of hex-coded bytes (e.g. '9A2C...')"""

    #Check that input paramaters

    if len(string)!=2*(1600)//8:
        print "string can't be divided in 25 blocks of w bits"
        exit(1)

    #Convert
    output=[[0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]]
    for x in range(5):
        for y in range(5):
            offset=2*((5*y+x)*64)//8
            output[x][y]=fromHexStringToLane(string[offset:offset+(2*64//8)])
    return output


def convertTableToStr(table):
    """Convert a 5×5 matrix representation to its string representation"""

    #Check input format
    if (len(table)!=5) or (False in [len(row)==5 for row in table]):
        print "table must be 5×5"
        exit(1)

    #Convert
    output=['']*25
    for x in range(5):
        for y in range(5):
            output[5*y+x]=fromLaneToHexString(table[x][y])
    output =''.join(output).upper()
    return output


def Keccak(input_str, n, program):


    P = pad10star1([len(input_str)*4, input_str],r) 
    print "Padded input"
    print P
    print

    # Initialisation of state
    S=[[0,0,0,0,0],
       [0,0,0,0,0],
       [0,0,0,0,0],
       [0,0,0,0,0],
       [0,0,0,0,0]]

    #Testing
    S = np.array([np.uint64(x) for x in S])

    for i in range((len(P)*8//2)//r):
        #print 'Passed into convertStrToTable:',P[i*(2*r//8):(i+1)*(2*r//8)]+'00'*(c//8)
        Pi=convertStrToTable(P[i*(2*r//8):(i+1)*(2*r//8)]+'00'*(c//8))

        for y in range(5):
            for x in range(5):
                #print 'type S:',type(S[x][y]),'type P:',type(Pi[x][y])
                S[x][y] = S[x][y]^Pi[x][y]

        S = np.array([np.uint64(x) for x in S])
        start = time.time()
        S = KeccakF(S, program)
        print "Time to run KeccakF: " + str(time.time() - start)
        #print S

    #Squeezing phase
    Z = ''
    outputLength = n
    while outputLength>0:
        string=convertTableToStr(S)
        Z = Z + string[:r*2//8]
        outputLength -= r
        if outputLength>0:
            S = KeccakF(S)

        # NB: done by block of length r, could have to be cut if outputLength
        #     is not a multiple of r

    return Z[:2*n//8]

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


    #PARAMETERS
    r = 576
    c = 1024
    n = 512


    to_hash= np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    to_hash2= np.array([[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    to_hash3= np.array([[1,1,1,1,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    inputnum = 3


    #print to_hash
    #print np.append(to_hash, to_hash2, axis=1)
    to_hash= np.append(to_hash, to_hash2, axis=0)
    to_hash = np.append(to_hash, to_hash3, axis=0) 
    #print to_hash
    start = time.time()
    S = KeccakF(to_hash, program)
    print "Time to run KeccakF: " + str(time.time() - start)

    to_hash = np.array([np.uint64(x) for x in to_hash])

    start = time.time()
    result = Keccak("A"*10000, n, program)
    print 'Convert str to table:',convertStrToTable("A"*16*25)
    print result
    print "Time taken is: " + str(time.time() - start)

###
    # original_str = np.array([["A"]*16 for x in range(25)])
    mf = cl.mem_flags
    # in_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=original_str)
    # out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=str_size)
    # copied_str = np.zeros_like(original_str)
###
    host_table = np.zeros((5,5))
    host_table = np.array([np.uint64(x) for x in host_table])  
    
    host_string = np.array(['A']*16*25)#,'B','C'])#]*16*25)
    #host_string = np.array([1,2,3])
    #Copy host string to gpu
    gpu_string = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_string)
    #cl.enqueue_copy(queue, gpu_string, host_string, is_blocking=False)

    print 'host_table before conversion:',host_table
    gpu_table = cl.Buffer(context, cl.mem_flags.READ_WRITE, 25*8)
#    program.convert_str_to_table(queue,(5,5),(5,5),gpu_string, gpu_table,5,5,64)
    program.convert_str_to_table(queue,(5,5),(5,5), gpu_string, gpu_table,np.uint64(5),np.uint64(5),np.uint64(64))
    cl.enqueue_copy(queue, host_table, gpu_table, is_blocking=True)
    print 'host_table after conversion:',host_table




