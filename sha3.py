from __future__ import division
import sys
import pyopencl as cl
import numpy as np
import pylab
import pdb

#Only pad for string messages not integers???
def pad10star1(self, M, n):
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

if __name__ == '__main__':
    # List our platforms
    platforms = cl.get_platforms()

    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices[:2])
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    
    program = cl.Program(context, open('sha3.cl').read()).build(options='')


    WORDLENGTH = 64
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

    to_hash= np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])

    to_hash = np.array([np.uint64(x) for x in to_hash])

    stuff_to_hash = cl.Buffer(context, cl.mem_flags.READ_ONLY, to_hash.size * 8)
    cl.enqueue_copy(queue, stuff_to_hash, to_hash, is_blocking=False)#is_block=True means wait for completion

    #Buffer for GPU to write final hash
    gpu_final_hash = cl.Buffer(context, cl.mem_flags.READ_WRITE, to_hash.size * 8)
    
    #Create 5x5 workgroup, local buffer
    local_size, global_size = (5, 5) , (5,5)
    local_buf_w,local_buf_h = np.uint64(5),np.uint64(5)


    gpu_local_memory = cl.LocalMemory(to_hash.size * 8)
    #A = cl.LocalMemory(to_hash.size * 8)
    #B = cl.LocalMemory(to_hash.size * 8)
    A = cl.LocalMemory(8*25)
    B = cl.LocalMemory(8*25)
    C = cl.LocalMemory(8*25)
    D = cl.LocalMemory(8*25)

    #Hash input
    final_hash = np.zeros((5,5))
    final_hash = np.array([np.uint64(x) for x in final_hash])    
    hash_event = program.sha_3_hash(queue, global_size, local_size,
                              stuff_to_hash, gpu_final_hash,rotation_gpu_buffer,round_constants_gpu,
                              B,A, C, D, local_buf_w,local_buf_h)

    
    
    cl.enqueue_copy(queue, final_hash, gpu_final_hash, is_blocking=True)

    #Profiling part
    seconds = (hash_event.profile.end - hash_event.profile.start) / 1e9
    print 'Total seconds to hash:',seconds
    #print final_hash
    hex_output = [map(hex, l) for l in np.transpose(final_hash)]
    print "output:"
    for x in range(len(hex_output)):
        print hex_output[x]

    #cl.Buffer = global memory
    #cl.LocalMemory = local memory

    # host_image = np.load('maze2.npy')
    # host_labels = np.empty_like(host_image)
    # host_done_flag = np.zeros(1).astype(np.int32)

    # gpu_image = cl.Buffer(context, cl.mem_flags.READ_ONLY, host_image.size * 4)
    # gpu_labels = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    # gpu_done_flag = cl.Buffer(context, cl.mem_flags.READ_WRITE, 4)
    #pdb.set_trace()
    # Send to the device, non-blocking
    # cl.enqueue_copy(queue, gpu_image, host_image, is_blocking=False)

    # global_size = tuple([round_up(g, l) for g, l in zip(host_image.shape[::-1], local_size)])
    # print global_size
    # width = np.int32(host_image.shape[1])
    # height = np.int32(host_image.shape[0])
    # halo = np.int32(1)
    
    # Create a local memory per working group that is
    # the size of an int (4 bytes) * (N+2) * (N+2), where N is the local_size
    # buf_size = (np.int32(local_size[0] + 2 * halo), np.int32(local_size[1] + 2 * halo))
    

    # initialize labels
    # program.initialize_labels(queue, global_size, local_size,
    #                           gpu_image, gpu_labels,
    #                           width, height)

    # while not done, propagate labels
    # itercount = 0

    # Show the initial labels
    # cl.enqueue_copy(queue, host_labels, gpu_labels, is_blocking=True)
    # pylab.imshow(host_labels)
    # pylab.title(itercount)
    # pylab.show()

    # show_progress = True
    # total_time = 0

    # while True:
    #     itercount += 1
    #     host_done_flag[0] = 0
    #     print 'iter', itercount
    #     cl.enqueue_copy(queue, gpu_done_flag, host_done_flag, is_blocking=False)
    #     prop_exec = program.propagate_labels(queue, global_size, local_size,
    #                                          gpu_labels, gpu_done_flag,
    #                                          gpu_local_memory,
    #                                          width, height,
    #                                          buf_size[0], buf_size[1],
    #                                          halo)
    #     prop_exec.wait()
    #     elapsed = 1e-6 * (prop_exec.profile.end - prop_exec.profile.start)
    #     total_time += elapsed
    #     # read back done flag, block until it gets here
    #     cl.enqueue_copy(queue, host_done_flag, gpu_done_flag, is_blocking=True)
    #     if host_done_flag[0] == 0:
    #         # no changes
    #         break
    #     # there were changes, so continue running
    #     print host_done_flag
    #     if itercount % 100 == 0 and show_progress:
    #         cl.enqueue_copy(queue, host_labels, gpu_labels, is_blocking=True)
    #         pylab.imshow(host_labels)
    #         pylab.title(itercount)
    #         pylab.show()
    #     if itercount % 10000 == 0:
    #         print 'Reached maximal number of iterations, aborting'
    #         sys.exit(0)

    # print('Finished after {} iterations, {} ms total, {} ms per iteration'.format(itercount, total_time, total_time / itercount))
    # # Show final result
    # cl.enqueue_copy(queue, host_labels, gpu_labels, is_blocking=True)
    # print 'Found {} regions'.format(len(np.unique(host_labels)) - 1)
    # pylab.imshow(host_labels)
    # pylab.title(itercount)
    # pylab.show()
