import Keccak
import sha3
import time
import pyopencl as cl

#convert a string input to ascii equivalent
def char_to_hex(input_chars):
	char_str = ""
	#-1 for range because the script contains a newline at the end
	for x in range(len(input_chars)):
		char_str +=str( format(ord(input_chars[x]),"x"))
	return char_str



#dictionary used

def run_serial():
	dictionary = open("john_the_ripper_dictionary.txt")
	output_to_file = open("output_serial.txt", "wb")

	myKeccak=Keccak.Keccak(1600)

	counter = 0

	start = time.time()
	while 1:
		line = dictionary.readline()
		if not line:
		    break
		counter = counter +1
		
		#convert a line in the dictionary to ascii
		#this is a newline at the end of line. remove it
		hexstring = char_to_hex(line[:-1])

		hash_result = myKeccak.Keccak([len(hexstring)*4,hexstring],r = 576, c=1024,n=512, verbose=False)
		#print hash_result
		#print
		output_to_file.write( hash_result + "\n");

	print "total time for Keccak Serial:" + str(time.time() - start)
	dictionary.close()
	output_to_file.close()

def run_parallel():
	dictionary = open("john_the_ripper_dictionary.txt")
	output_to_file = open("output_parallel.txt", "wb")

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
	print 'The queue is using the device:', queue.device.name


	program = cl.Program(context, open('sha3.cl').read()).build(options='')


	#PARAMETERS for SHA 512
	r = 576
	c = 1024
	n = 512


	counter = 0


	start = time.time()
	inputlist = []
	while 1:
		line = dictionary.readline()
		if not line:
		    break
		counter = counter +1
		
		#convert a line in the dictionary to ascii
		#this is a newline at the end of line. remove it
		hexstring = char_to_hex(line[:-1])

		inputlist.append(hexstring)

	result = sha3.Keccak(inputlist, n, r, c, program, context, queue)
	#print  "Hashing Result is"
	for x in range(len(result)):
		output_to_file.write( result[x] + "\n");

	print "total time for Keccak parallel:" + str(time.time() - start)
	# Close opend file
	dictionary.close()
	output_to_file.close()



if __name__ == '__main__':
	run_serial()
	run_parallel()