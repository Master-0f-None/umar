// TODO(umar) move out of here
#define ZCASH_BLOCK_HEADER_LEN 140
#define ZCASH_HASH_LEN 50

ulong8 blake_iv = (ulong8)(
  0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
  0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
  0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
);

/*
** If xi0,xi1,xi2,xi3 are stored consecutively in little endian then they
** represent (hex notation, group of 5 hex digits are a group of PREFIX bits):
**   aa aa ab bb bb cc cc cd dd...  [round 0]
**         --------------------
**      ...ab bb bb cc cc cd dd...  [odd round]
**               --------------
**               ...cc cc cd dd...  [next even round]
**                        -----
** Bytes underlined are going to be stored in the slot. Preceding bytes
** (and possibly part of the underlined bytes, depending on NR_ROWS_LOG) are
** used to compute the row number.
**
** Round 0: xi0,xi1,xi2,xi3 is a 25-byte Xi (xi3: only the low byte matter)
** Round 1: xi0,xi1,xi2 is a 23-byte Xi (incl. the colliding PREFIX nibble)
** TODO: update lines below with padding nibbles
** Round 2: xi0,xi1,xi2 is a 20-byte Xi (xi2: only the low 4 bytes matter)
** Round 3: xi0,xi1,xi2 is a 17.5-byte Xi (xi2: only the low 1.5 bytes matter)
** Round 4: xi0,xi1 is a 15-byte Xi (xi1: only the low 7 bytes matter)
** Round 5: xi0,xi1 is a 12.5-byte Xi (xi1: only the low 4.5 bytes matter)
** Round 6: xi0,xi1 is a 10-byte Xi (xi1: only the low 2 bytes matter)
** Round 7: xi0 is a 7.5-byte Xi (xi0: only the low 7.5 bytes matter)
** Round 8: xi0 is a 5-byte Xi (xi0: only the low 5 bytes matter)
**
** Return 0 if successfully stored, or 1 if the row overflowed.
*/
uint ht_store_transposed(__global ulong4 *ht, uint i, ulong4 xi, __global atomic_uint *bucket_counts) {

  // TODO(umar): these should be a static values
  int bit_string_len = sizeof(ulong4); // number of bytes in the bit_string 25 == 200
  int num_bins = 1 << 20;              // number of bins 1M
  ulong mask = 0xfffff00000000000;    // first twenty bits

  uint bin = (xi[0] & mask) >> (64 - 20);

  int bin_comp_index = (bin / 8);
  int bin_comp_shift = bin % 8;

  // Increment the count of the bin. The return value is the index within the bin
  uint bin_index = atomic_fetch_add_explicit(bucket_counts+bin_comp_index, 1 << bin_comp_shift, memory_order_relaxed, memory_scope_device);

  uint loc = (bin_index >> bin_comp_shift) * num_bins + bin;

  ht[loc] = xi;
}

uint ht_store(__global ulong4 *ht, uint i, ulong4 xi, __global atomic_uint *bucket_counts, const int num_bins) {

  // TODO(umar): these should be a static values
  static int bit_string_len = sizeof(ulong4); // number of bytes in the bit_string 25 == 200
  static ulong mask = 0xfffff00000000000;    // first twenty bits

  const uint bucket = (xi[0] & mask) >> (64 - 20);

  int bucket_comp_index = (bucket / 8);
  int bucket_comp_shift = (bucket % 8) * 4;

  // Increment the count of the bin. The return value is the index within the bin
  const uint bin_index = atomic_fetch_add_explicit(bucket_counts+bucket_comp_index, (1u << bucket_comp_shift), memory_order_relaxed, memory_scope_device);
  const uint loc =  bucket * num_bins + ((bin_index >> bucket_comp_shift) & 0xf);

  ht[loc] = xi;
}

#define mix(va, vb, vc, vd, x, y)               \
  va = (va + vb + x);                           \
  vd = rotate((vd ^ va), (ulong)64 - 32);       \
  vc = (vc + vd);                               \
  vb = rotate((vb ^ vc), (ulong)64 - 24);       \
  va = (va + vb + y);                           \
  vd = rotate((vd ^ va), (ulong)64 - 16);       \
  vc = (vc + vd);                               \
  vb = rotate((vb ^ vc), (ulong)64 - 63);
/*
** Note: making the work group size less than or equal to the wavefront size
** allows the OpenCL compiler to remove the barrier() calls, see "2.2 Local
** Memory (LDS) Optimization 2-10" in:
**
*http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/opencl-optimization-guide/
*/
__kernel
void
blake(__global ulong *bs, __global ulong4 *ht,
      __global atomic_uint *bucket_count,
      const int num_bins) {
  uint tid = get_global_id(0);
  uint inputs_per_thread = 1;// / get_global_size(0);
  uint input = tid * inputs_per_thread;
  uint input_end = (tid + 1) * inputs_per_thread;
  __local ulong blake_state[8];
  event_t event = async_work_group_copy(blake_state, bs, 8, 0);

    // shift "i" to occupy the high 32 bits of the second ulong word in the
    // message block
    ulong word1 = (ulong)input << 32;

    // init vector v
    wait_group_events(1, &event);
    ulong16 v = (ulong16)(*(ulong8*)blake_state, blake_iv);
    v[12] ^= ZCASH_BLOCK_HEADER_LEN + 4 /* length of "i" */;
    v[14] ^= (ulong)-1;

    // round 1
    mix(v[0], v[4], v[8], v[12], 0, word1);
    mix(v[1], v[5], v[9], v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8], v[13], 0, 0);
    mix(v[3], v[4], v[9], v[14], 0, 0);
    // round 2
    mix(v[0], v[4], v[8], v[12], 0, 0);
    mix(v[1], v[5], v[9], v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], word1, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8], v[13], 0, 0);
    mix(v[3], v[4], v[9], v[14], 0, 0);
    // round 3
    mix(v[0], v[4], v[8], v[12], 0, 0);
    mix(v[1], v[5], v[9], v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8], v[13], 0, word1);
    mix(v[3], v[4], v[9], v[14], 0, 0);
    // round 4
    mix(v[0], v[4], v[8], v[12], 0, 0);
    mix(v[1], v[5], v[9], v[13], 0, word1);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8], v[13], 0, 0);
    mix(v[3], v[4], v[9], v[14], 0, 0);
    // round 5
    mix(v[0], v[4], v[8], v[12], 0, 0);
    mix(v[1], v[5], v[9], v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, word1);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8], v[13], 0, 0);
    mix(v[3], v[4], v[9], v[14], 0, 0);
    // round 6
    mix(v[0], v[4], v[8], v[12], 0, 0);
    mix(v[1], v[5], v[9], v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8], v[13], 0, 0);
    mix(v[3], v[4], v[9], v[14], word1, 0);
    // round 7
    mix(v[0], v[4], v[8], v[12], 0, 0);
    mix(v[1], v[5], v[9], v[13], word1, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8], v[13], 0, 0);
    mix(v[3], v[4], v[9], v[14], 0, 0);
    // round 8
    mix(v[0], v[4], v[8], v[12], 0, 0);
    mix(v[1], v[5], v[9], v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, word1);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8], v[13], 0, 0);
    mix(v[3], v[4], v[9], v[14], 0, 0);
    // round 9
    mix(v[0], v[4], v[8], v[12], 0, 0);
    mix(v[1], v[5], v[9], v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8], v[13], word1, 0);
    mix(v[3], v[4], v[9], v[14], 0, 0);
    // round 10
    mix(v[0], v[4], v[8], v[12], 0, 0);
    mix(v[1], v[5], v[9], v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], word1, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8], v[13], 0, 0);
    mix(v[3], v[4], v[9], v[14], 0, 0);
    // round 11
    mix(v[0], v[4], v[8], v[12], 0, word1);
    mix(v[1], v[5], v[9], v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8], v[13], 0, 0);
    mix(v[3], v[4], v[9], v[14], 0, 0);
    // round 12
    mix(v[0], v[4], v[8], v[12], 0, 0);
    mix(v[1], v[5], v[9], v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], word1, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8], v[13], 0, 0);
    mix(v[3], v[4], v[9], v[14], 0, 0);

    // compress v into the blake state; this produces the 50-byte hash
    // (two Xi values)
    ulong8 h;
    h[0] = blake_state[0] ^ v[0] ^ v[8];
    h[1] = blake_state[1] ^ v[1] ^ v[9];
    h[2] = blake_state[2] ^ v[2] ^ v[10];
    h[3] = blake_state[3] ^ v[3] ^ v[11];
    h[4] = blake_state[4] ^ v[4] ^ v[12];
    h[5] = blake_state[5] ^ v[5] ^ v[13];
    h[6] = (blake_state[6] ^ v[6] ^ v[14]) & 0xffff;

    // store the two Xi values in the hash table
    ulong4 first_bitstring = (ulong4)(h.s012, h[3] >> (64 - 8) << (64 - 8));
    ulong4 second_bitstring = (ulong4)((h[3] << 8) | (h[4] >> (64 - 8)),
                                      (h[4] << 8) | (h[5] >> (64 - 8)),
                                      (h[5] << 8) | (h[6] >> (8)),
                                       (h[6] & 0xff) << (64-8));

    ht_store(ht, input * 2,     first_bitstring,  bucket_count, num_bins);
    ht_store(ht, input * 2 + 1, second_bitstring, bucket_count, num_bins);

    input++;
}
