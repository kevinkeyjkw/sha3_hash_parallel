//#include <stdlib.h>
#define YES 1
#define NO  0
 
ulong hexToInt(char s[]) {
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
                        __local ulong *B, 
                        __local ulong *A, __local ulong *C, __local ulong *D, ulong buf_w, ulong buf_h){

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const int x = get_global_id(0);
    const int y = get_global_id(1);    

    const ulong wordlength = 64;

    //Each thread responsible for loading its value from global to local
    A[ly*buf_w+lx] = original_hash[y*buf_w+x];
    //Make sure threads have finished loading local buffer
    barrier(CLK_LOCAL_MEM_FENCE);
    //Assume have B(5x5) and rotation offsets(5x5)

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

    //Write A to global
    final_hash[y * buf_w + x] = A[ly * buf_w + lx];
    barrier(CLK_LOCAL_MEM_FENCE);    
}

__kernel void convert_str_to_table(__global __read_only char *string_to_convert,
                        __global __write_only ulong *table, ulong buf_w, ulong buf_h, ulong lane_bit_size){
//__kernel void convert_str_to_table(__global __write_only ulong *table, ulong buf_w, ulong buf_h, ulong lane_bit_size){
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    //Offset into string
    //printf("Testing");
    ulong offset = (5 * lx + ly)* lane_bit_size / 4;
    //Store the part of string to convert

    char part_of_string[16];
    //printf("%c",string_to_convert[1]);
    //Copy 16 hex characters (64 bits) from large string into another variable
    int k=0;
    int l = offset;
    while(k<lane_bit_size/4){
        part_of_string[k] = string_to_convert[l];
        k+=1;
        l+=1;
    }
    //printf("%c ",part_of_string[0]);
    //printf("%d ",sizeof(part_of_string));
    // //Convert that part of the string from hex characters to int 
    //     //1. Convert 'AB CD EF GH' to 'GH EF CD AB'
        int i=0;
        int j=sizeof(part_of_string)-2;
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

    // //2. Convert hex string to int and store in table
    //printf("%lu ",hexToInt(part_of_string));
    table[ly*buf_w + lx] = hexToInt(part_of_string);//strtol(part_of_string, 0, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
}




// int rotate(int toRotate,int rotate_offset){
//     return toRotate >> rotate_offset;
// }
// __kernel void
// initialize_labels(__global __read_only int *image,
//                   __global __write_only int *labels,
//                   int w, int h)
// {
//     const int x = get_global_id(0);
//     const int y = get_global_id(1);

//     if ((x < w) && (y < h)) {
//         if (image[y * w + x] > 0) {
//             // set each pixel > 0 to its linear index
//             labels[y * w + x] = y * w + x;
//         } else {
//             // out of bounds, set to maximum
//             labels[y * w + x] = w * h;
//         }
//     }
// }

// int
// get_clamped_value(__global __read_only int *labels,
//                   int w, int h,
//                   int x, int y)
// {
//     if ((x < 0) || (x >= w) || (y < 0) || (y >= h))
//         return w * h;
//     return labels[y * w + x];
// }

// __kernel void
// propagate_labels(__global __read_write int *labels,
//                  __global __write_only int *changed_flag,
//                  __local int *buffer,
//                  int w, int h,
//                  int buf_w, int buf_h,
//                  const int halo)
// {
//     // halo is the additional number of cells in one direction

//     // Global position of output pixel
//     const int x = get_global_id(0);
//     const int y = get_global_id(1);

//     // Local position relative to (0, 0) in workgroup
//     const int lx = get_local_id(0);
//     const int ly = get_local_id(1);

//     // coordinates of the upper left corner of the buffer in image
//     // space, including halo
//     const int buf_corner_x = x - lx - halo;
//     const int buf_corner_y = y - ly - halo;

//     // coordinates of our pixel in the local buffer
//     const int buf_x = lx + halo;
//     const int buf_y = ly + halo;

//     // 1D index of thread within our work-group
//     const int idx_1D = ly * get_local_size(0) + lx;
    
//     int old_label;
//     // Will store the output value
//     int new_label;
    
//     // Load the relevant labels to a local buffer with a halo 
//     if (idx_1D < buf_w) {
//         for (int row = 0; row < buf_h; row++) {
//             buffer[row * buf_w + idx_1D] = 
//                 get_clamped_value(labels,
//                                   w, h,
//                                   buf_corner_x + idx_1D, buf_corner_y + row);
//         }
//     }

//     // Make sure all threads reach the next part after
//     // the local buffer is loaded
//     barrier(CLK_LOCAL_MEM_FENCE);

//     // Fetch the value from the buffer that corresponds to
//     // the pixel for this thread
//     old_label = buffer[buf_y * buf_w + buf_x];

//     // CODE FOR PARTS 2 and 4 HERE (part 4 will replace part 2)
//     //Part 2
//     // if(x >= 0 && x < w && y >= 0 && y < h){
//     //     if(old_label<w*h){
//     //         buffer[buf_y*buf_w+buf_x] = labels[old_label];
//     //     }
//     // }
//     //Part 4
//     //Let the first thread of each workgroup do the updates
//     if(lx+ly==0){
//         int prev_idx = old_label;
//         int prev_val = labels[prev_idx];
//         //Loop through the local buffer
//         for(int c=0;c < buf_w*buf_h;c++){
//             //Only updates pixels of wall
//             if(prev_idx < w*h){
//                 if(buffer[c] != prev_idx){
//                     prev_idx = buffer[c];
//                     prev_val = labels[prev_idx];
//                 }
//                 //Replace with label of label
//                 buffer[c] = prev_val;
//             }
//         }
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);
    
//     // stay in bounds
//     if ((x < w) && (y < h)) {
//         // CODE FOR PART 1 HERE
//         // We set new_label to the value of old_label, but you will need
//         // to adjust this for correctness.

//         //Take the min of neighbors(left,right,top,bottom) only if pixel is a wall(value less than w*h)
//         if(old_label<w*h){
//             new_label = min(min(min(min(buffer[(buf_y-1)*buf_w+buf_x],\
//                 buffer[(buf_y)*buf_w+buf_x-1]),buffer[(buf_y)*buf_w+buf_x+1]),\
//                 buffer[(buf_y+1)*buf_w+buf_x]),old_label);
//         }else{
//             //value of label did not change
//             new_label=old_label;
//          }

//         if (new_label != old_label) {
//             // CODE FOR PART 3 HERE
//             // indicate there was a change this iteration.
//             // multiple threads might write this.
//             // *(changed_flag) += 1;
//             // labels[y * w + x] = new_label;
//             *(changed_flag) += 1;
//             atomic_min(labels + old_label,new_label);
//             atomic_min(labels+(y * w) + x, new_label);
//             //Part 5 use min() instead of atomic_min()
//             //*(labels + old_label) = min(*(labels+old_label),new_label);
//             //*(labels +(y*w)+x) = min(*(labels +(y*w)+x),new_label);
//         }
//     }
// }
