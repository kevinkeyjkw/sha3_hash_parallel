#define YES 1
#define NO  0
 
ulong hexToInt(__local char s[]) {
    int hexdigit, i, inhex;
    ulong n;    
    i=0;
     
    if(s[i] == '0') {
        ++i;
        if(s[i] == 'x' || s[i] == 'X'){            
            ++i;
        }
    }
     
    n = 0;
    inhex = YES;
     //for(; inhex == YES; ++i) {
    while(inhex == YES) {
        if(s[i] >= '0' && s[i] <= '9') {            
            hexdigit = s[i] - '0';
        } else if(s[i] >= 'a' && s[i] <= 'f') {            
            hexdigit = s[i] - 'a' + 10;
        } else if(s[i] >= 'A' && s[i] <= 'F') {
            hexdigit = s[i] - 'A' + 10;
        } else {
            inhex = NO;
        }
         
        if(inhex == YES) {
            n = 16 * n + hexdigit;
        }
        i+=1;
    }
     
    return n;
}
//rotate input by x bit to the left, where input is of bitlength size
ulong rotateFunction(ulong input, ulong bits, ulong bitlength)
{
    if (bits == 0)
    {
        return input;
    }

        bits = bits%bitlength;
        return ((input>>(bitlength-bits))+(input<<bits));
}

__kernel void sha_3_hash(__global __read_only ulong *original_hash,
                        __global __write_only ulong *final_hash,
                        __global __read_only ulong *rotation_offsets,
                        __global __read_only ulong *RCfixed,
                        __global __read_only ulong *iterations,
                        __global __read_only ulong  *curr_iteration,
                        __local ulong *B, 
                        __local ulong *A, __local ulong *C, __local ulong *D, ulong buf_w, ulong buf_h){

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const int x = get_global_id(0);
    const int y = get_global_id(1);    

    const int group_id = get_group_id(1);

    const ulong wordlength = 64;

    //Each thread responsible for loading its value from global to local
    A[ly*buf_w+lx] = original_hash[y*buf_w+x];
    //Make sure threads have finished loading local buffer
    barrier(CLK_LOCAL_MEM_FENCE);
    //Assume have B(5x5) and rotation offsets(5x5)

/*
    if(x==0 && y==0){
        printf("iterations %d\n", iterations[group_id]);
        //printf("iterations %d\n", iterations[3]);
        //printf("iterations %d\n", iterations[4]);
        printf("curr iteration %d\n", curr_iteration[0]);
    }
*/

    if (curr_iteration[0] < iterations[group_id])
    {

        for (int roundcounter = 0; roundcounter < 24; roundcounter++)
        {

            /*
            if(lx==0 && ly==0){
                printf("Starting Round:%i: %i global:%i, %i, %lu\n", roundcounter, lx, x,y, A[1]);
            }
            */
            //Theta step
            C[lx] = A[lx*5]^A[lx*5+1]^A[lx*5+2]^A[lx*5+3]^A[lx*5+4]; 
            //Dual xor lane
            D[lx] =  C[(lx+4)%5]^rotateFunction(C[(lx+1)%5],1, wordlength);
            barrier(CLK_LOCAL_MEM_FENCE);
            A[ly*buf_w+lx] = A[ly*buf_w+lx] ^ D[ly];   
            barrier(CLK_LOCAL_MEM_FENCE);
            //Rho step
            //Pi step
            B[lx * buf_w + ((2 * ly + 3 * lx) % 5)] = rotateFunction(
                                    A[ly * buf_w + lx],
                                    rotation_offsets[ly * buf_w + lx], wordlength);
            barrier(CLK_LOCAL_MEM_FENCE);
            //Chi step
            A[ly * buf_w + lx] = B[ly * buf_w + lx] ^ (
                (~B[((ly+1) % 5) * buf_w + lx]) & 
                B[((ly+2)%5) * buf_w + lx]
                );
            barrier(CLK_LOCAL_MEM_FENCE);
            //Iota step, Used RCfixed which depends on round number
            if(lx==0 && ly==0){
                A[0] = A[0] ^ RCfixed[roundcounter];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    //Write A to global
    final_hash[y * buf_w + x] = A[ly * buf_w + lx];
    barrier(CLK_LOCAL_MEM_FENCE);    
}

__kernel void convert_str_to_table(__global __read_only char *string_to_convert,
                        __global __write_only ulong *table,
                        __local char *part_of_string, 
                         ulong buf_w, ulong buf_h, ulong lane_bit_size){
//__kernel void convert_str_to_table(__global __write_only ulong *table, ulong buf_w, ulong buf_h, ulong lane_bit_size){
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const int x = get_global_id(0);
    const int y = get_global_id(1);  


    const int group_id = get_group_id(1);
    //Offset into string




    ulong offset = (5 * lx + ly)* lane_bit_size / 4;
    //Store the part of string to convert


    //Copy 16 hex characters (64 bits) from large string into another variable
    int k=0;
    int l = offset;
    while(k<lane_bit_size/4){
        part_of_string[k] = string_to_convert[l + group_id*400];
        k+=1;
        l+=1;
    }


    // //Convert that part of the string from hex characters to int 
    //     //1. Convert 'AB CD EF GH' to 'GH EF CD AB'
    int i=0;
    //int j=sizeof(part_of_string)-2;
    int j = 14;

    while(i <= 6){
        char tmpA = part_of_string[i];
        char tmpB = part_of_string[i+1];
        part_of_string[i] = part_of_string[j];
        part_of_string[i+1] = part_of_string[j+1];
        part_of_string[j] = tmpA;
        part_of_string[j+1] = tmpB;
        i += 2;
        j -= 2;
    }

/*
    if(lx==0 && ly==0){
        //printf("%d\n",group_id);
        for (int a = 0; a< 16; a++)
        {
            printf("%c\n",part_of_string[a]);
      
        }
    }
*/

    // //2. Convert hex string to int and store in table
    //printf("%lu ",hexToInt(part_of_string));
    table[y*buf_w + x] = hexToInt(part_of_string);//strtol(part_of_string, 0, 16);


    barrier(CLK_LOCAL_MEM_FENCE);
}


__kernel void convert_table_to_str(__global __read_only ulong *table,
                        __global __write_only char *output_str, ulong buf_w, ulong buf_h, ulong lane_bit_size){
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const int x = get_global_id(0);
    const int y = get_global_id(1);  

    const int group_id = get_group_id(1);

    unsigned long clearbits = 0x000000000000000f;
    int offset = 0;
    for (int a = 0; a< 8; a++)
    {
        //part_of_string[x] = (table[ly*buf_w + lx] & clearbits) >> offset;

        //(group_id *5 + lx)*buf_w + ly is necessary as we are taking the transpose of each workgroup
        output_str[(y*buf_w + x)*16 + a*2+1] = (table[(group_id *5 + lx)*buf_w + ly] & clearbits) >> offset;
        clearbits = clearbits << 4;
        offset = offset + 4;
        output_str[(y*buf_w + x)*16 + a*2] = (table[(group_id *5 + lx)*buf_w + ly] & clearbits) >> offset;
        clearbits = clearbits << 4;
        offset = offset + 4;


    }

    for (int a = 0; a<16; a++)
    {
        if (output_str[(y*buf_w + x)*16 + a] > 9)
        {
            output_str[(y*buf_w + x)*16 + a] = output_str[(y*buf_w + x)*16 + a] + 55;
        }
        else
        {
            output_str[(y*buf_w + x)*16 + a] = output_str[(y*buf_w + x)*16 + a] + 48;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    /*
    if(lx==0 && ly==0){
            printf("printing x and y %d %d", x,y );         
            printf("printing %lx\n", (unsigned long)table[x*buf_w + y] );
            //printf("printing %lx\n", (unsigned long)table[ly*buf_w + lx] & 0x00000000000000ff);
            //printf("printing %lx\n", (unsigned long)table[ly*buf_w + lx] & 0x000000000000ff00);

            printf("print: %c\n", output_str[(ly*buf_w + lx)*16 + 0]);
            printf("print: %c\n", output_str[(ly*buf_w + lx)*16 + 1]);
            printf("print: %c\n", output_str[(ly*buf_w + lx)*16 + 2]);
            
    }
    */

}
