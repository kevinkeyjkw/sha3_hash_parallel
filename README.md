# sha3_hash_parallel

###Setting Up
Our parallelized sha3 code requires an installation of OpenCL. This can be done via the following steps:
1. sudo apt-get install ocl-icd-opencl-dev
2. install AMD APP SDK at http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/
3. sudo apt-get install python-pyopencl

###Files
The serial code is taken from the authors of Keccak at http://keccak.noekeon.org/
The authors' serial code is in: Keccak.py
Our parallelized code is in: sha3.cl, sha3.py
Demo code given by the authors are: demo_Keccak.py (serial), demo_KeccakF.py (serial)
Our runtime comparison script is in: keccak_performancetest.py
Dictionary used in our project are taken from "John the Ripper" and "Cain & Abel"

###Running SHA3
Our script takes in a list of strings and hash the list in parallel. It can be run in the following manner:

```python
	import sha3

	#Initialize OpenCL
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()
    context = cl.Context(devices[:2])
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    program = cl.Program(context, open('sha3.cl').read()).build(options='')

    #Parameters for SHA 512
    r = 576
    c = 1024
    n = 512

    #Input strings
    inputlist = []
    inputlist.append("")
    inputlist.append("abcd")
    inputlist.append("a" * 1000)

    #Get results
    result = sha3.Keccak(inputlist, n, r,c, program, context, queue)
```

The output will be a list of hashes that correspond to the input strings.