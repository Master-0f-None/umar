

#define LOCAL_SIZE 128
#define SUB_GROUP_SIZE 64
#define NUM_SUB_GROUPS LOCAL_SIZE / SUB_GROUP_SIZE

int solve(const int i) {
  float d = 1 + 8*i;
  float k = -1 + native_sqrt(d);
  k = k/2;
  k = ceil(k);
  return k;
}

int2 get_pair(const int i) {
  int k = solve(i);
  int start = (k*(k-1)/2);
  return (int2)( k+1, i - start);
}

int pair_count(int collisions) {
  switch(collisions) {
  case 2: return 1;
  case 3: return 3;
  case 4: return 6;
  case 5: return 10;
  case 6: return 15;
  case 7: return 21;
  case 8: return 28;
  case 9: return 36;
  case 10: return 45;
  case 11: return 55;
  case 12: return 66;
  case 13: return 78;
  default: return 0;
  }
}

#define BITSTRING_BUFFER_SIZE 48
#define COLLISION_DEPTH 12


// Executes a round of the equihash function and stores the results into the dst_ht
// hash table
//
// \param[in]  hash_mask         The mask used to determine the bucket in the dst_ht
// \param[out] dst_ht            The destination hash table pointer
// \param[out] dst_bucket_counts The collision counts for each of the buckets in dst_ht
// \param[in]  src_ht            The source hash table
// \param[in]  src_bucket_counts The collision counts for each of the buckets in src_ht
// \param[in]  collision_size    The collion count this kernel searches for
kernel __attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
void equihash_round(global       ulong4 * const restrict dst_ht, global       atomic_uint * const restrict dst_bucket_counts,
                    global const ulong * const restrict src_ht, global const uint        * const restrict src_bucket_counts,
                    ulong4 prev_hash_mask,
                    ulong4 hash_mask,
                    int4 hash_mask_shift,
                    uint num_buckets,
                    uint collision_size) {
  local int   wg_collision_offsets [NUM_SUB_GROUPS][SUB_GROUP_SIZE];
  local uint  wg_bucket_index[NUM_SUB_GROUPS][SUB_GROUP_SIZE];
  local ulong wg_bitstring_store[NUM_SUB_GROUPS][BITSTRING_BUFFER_SIZE*4];
  local int   wg_pair_index[NUM_SUB_GROUPS][SUB_GROUP_SIZE];
  local atomic_uint wg_batch_offsets[NUM_SUB_GROUPS];

  local int* collision_offsets = wg_collision_offsets[get_sub_group_id()];
  local atomic_uint* batch_offsets = &(wg_batch_offsets[get_sub_group_id()]);
  local uint* bucket_index = wg_bucket_index[get_sub_group_id()];
  local ulong* bitstring_store = wg_bitstring_store[get_sub_group_id()];
  local int* pair_index = wg_pair_index[get_sub_group_id()];

  collision_offsets[get_sub_group_local_id()] = 0;
  bucket_index[get_sub_group_local_id()] = 0;
  __attribute__((opencl_unroll_hint(1)))
    for(int i = get_sub_group_local_id(); i < BITSTRING_BUFFER_SIZE; i += get_sub_group_size()) {
    ((ulong4*)bitstring_store)[i] = -1;
  }
  atomic_init(batch_offsets, 0);
  sub_group_barrier(CLK_LOCAL_MEM_FENCE);

  // each integer represents 8 counts encoded as 4 bits each
  if(get_sub_group_local_id() < 16)
  {
    int index = ((get_group_id(0) * NUM_SUB_GROUPS + get_sub_group_id()) * 16) + get_sub_group_local_id();
    int bucket_count_index = index >> 3;
    int bucket_count_shift = (index & 0x7) * 4;
    int count = (src_bucket_counts[bucket_count_index] >> bucket_count_shift) & 0xf;
    int lidx = sub_group_scan_inclusive_add(count >= collision_size)-1;
    if(count >= collision_size){
      collision_offsets[lidx] = count;
      bucket_index[lidx] = index;
    }
  }
  sub_group_barrier(CLK_LOCAL_MEM_FENCE);
  collision_offsets[get_sub_group_local_id()] = sub_group_scan_exclusive_add(collision_offsets[get_sub_group_local_id()]);

  // while the batch offset does not go past the end
  while(sub_group_any(atomic_load_explicit(batch_offsets, memory_order_relaxed, memory_scope_sub_group) < collision_offsets[SUB_GROUP_SIZE - 1])) {
    int num_pairs = 0;
    int offset = atomic_load_explicit(batch_offsets, memory_order_relaxed, memory_scope_sub_group);
    bool cond = get_sub_group_local_id() != SUB_GROUP_SIZE - 1     &&
      collision_offsets[get_sub_group_local_id()] >= offset    &&
      collision_offsets[get_sub_group_local_id() + 1] < (offset + BITSTRING_BUFFER_SIZE);
    if(cond) {
      //if(get_group_id(0) == 0 && get_sub_group_id() == 0) printf("%d %d", get_sub_group_local_id(), collision_offsets[get_sub_group_local_id()]);
      atomic_fetch_max_explicit(batch_offsets, collision_offsets[get_sub_group_local_id()+1], memory_order_relaxed, memory_scope_sub_group);

      char collision_count = collision_offsets[get_sub_group_local_id()+1] - collision_offsets[get_sub_group_local_id()];
      //if(get_global_id(0)<64) printf("%d %d %d", collision_count, collision_offsets[get_sub_group_local_id()], bucket_index[get_sub_group_id()][get_sub_group_local_id()]);
      for(int i = 0; i < collision_count; i++) {
        ((ulong4*)bitstring_store)[collision_offsets[get_sub_group_local_id()] + i - offset] = (ulong4)(bucket_index[get_sub_group_local_id()] * 4 * COLLISION_DEPTH + i * 4);
      }
      num_pairs = pair_count(collision_count);
    }
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);

    int end_offset = (atomic_load_explicit(batch_offsets, memory_order_relaxed, memory_scope_sub_group) - offset) * 4;
    // retrieve the bitstrings from global memory
    for(int i = get_sub_group_local_id(); i < end_offset; i+= SUB_GROUP_SIZE) {
      bitstring_store[i] = src_ht[bitstring_store[i] + (get_sub_group_local_id() & 0x3)];
    }
    pair_index[get_sub_group_local_id()] = sub_group_scan_inclusive_add(num_pairs);
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);

    for(int sgid = get_sub_group_local_id(); sub_group_any(sgid < pair_index[SUB_GROUP_SIZE - 1]); sgid += SUB_GROUP_SIZE) {
      int my_pair_index = -1;
      int diff = 0;
      int i = 0;
      while(pair_index[i] <= sgid && i < SUB_GROUP_SIZE ) {
        diff += pair_index[i] != 0;
        i++;
        if(pair_index[i] != 0 && pair_index[i] == pair_index[i-1]) break;
      }

      if(pair_index[i] > sgid) {
        my_pair_index = i;
      }

      if(my_pair_index != -1) {
        ulong4 result = 0;
        {
          int idx = collision_offsets[my_pair_index] - collision_offsets[my_pair_index-diff];
          int2 lbs_index = get_pair(pair_index[my_pair_index] - sgid) + idx - 1;
          result = ((ulong4*)bitstring_store)[lbs_index.x] ^ ((ulong4*)bitstring_store)[lbs_index.y];
        }

        if(result.s0 || result.s1 || result.s2 || result.s3) {
          ulong4 bucket4 = (hash_mask & result) >> hash_mask_shift;
          //bucket4 ^= (hash_mask & result) << -hash_mask_shift;
          int bucket = bucket4.x ^ bucket4.y ^ bucket4.z ^ bucket4.w;
          //if(get_global_id(0) == 0) printf("[%016v4lx]  [%016v4lx] | [%016lx]", result, bucket4, bucket);

          int bucket_comp_index = (bucket >> 3);
          int bucket_comp_shift = (bucket & 0x7) * 4;

          int bin_index = atomic_fetch_add_explicit(dst_bucket_counts+bucket_comp_index, 1u << bucket_comp_shift, memory_order_relaxed, memory_scope_device);
          dst_ht[bucket * COLLISION_DEPTH + ((bin_index >> bucket_comp_shift) & 0xf)] = result;
        }
      }
    }
  }

  return;
}


// This was an early attempt at the equihash round. It works on a transposed hash table and

kernel __attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
void equihash_round_transposed_ht(global ulong4 * restrict dst_ht, global atomic_uint * restrict dst_bucket_counts,
                    global const ulong4 * restrict src_ht, global const uint * restrict src_bucket_counts,
                    ulong4 hash_mask,
                    int4 hash_mask_shift,
                    uint num_buckets,
                    uint collision_size) {
  local int scan_index[NUM_SUB_GROUPS][SUB_GROUP_SIZE];
  local int scan_counts[NUM_SUB_GROUPS][SUB_GROUP_SIZE];

  local atomic_uint mask_bits[LOCAL_SIZE/4];

  const int lid_div_4 = get_local_id(0)/4;

  atomic_init(mask_bits + (lid_div_4), 0);
  scan_index[get_sub_group_id()][get_sub_group_local_id()] = -1;
  sub_group_barrier(CLK_LOCAL_MEM_FENCE);

  {
    uint count = src_bucket_counts[get_global_id(0)];

    int lidx = sub_group_scan_inclusive_add(count >= collision_size);
    if(count >= collision_size) {
      scan_index[get_sub_group_id()][lidx] = get_global_id(0);
      scan_counts[get_sub_group_id()][lidx] = count;
    }
  }

  // 2:1 3:3 4:6 5:10
  // a b
  //   ab
  // a b c
  //   ab ac bc
  // a b c d
  //   ab ac ad bc bd cd
  // a b c d e
  //   ab ac ad ae bc bd be cd ce de

  // (284032*2+189134*3+94189*4+37877*5+12769*6+3679*7+917*8+171*9+24*10+4*11+4*12) * 8 * 4 = 58,021,792 B

  // loop over all scan_index values
  const int bsid = get_sub_group_local_id() & 0x3;
  for(int scan_id = (get_sub_group_local_id() /4); scan_id < get_sub_group_size(); scan_id += (get_sub_group_size()/4)) {
    // 1. each thread brings in one ulong from src_ht
    ulong result = 0;
    ulong bucket;

    const int index = scan_index[get_sub_group_id()][scan_id];
    if(index != -1) {
      if(scan_counts[get_sub_group_id()][scan_id] > 1) {
        //result = ((global const ulong*)(src_ht + index))[bsid] ^ ((global const ulong*)(src_ht + num_buckets + index))[bsid];
        result = ((global const ulong*)(src_ht + index))[bsid] ^ ((global const ulong*)(src_ht + index + 1))[bsid];
      }

      bucket = (hash_mask[bsid] & result) >> hash_mask_shift[bsid];
      if(bucket) atomic_fetch_xor(mask_bits + (lid_div_4), bucket);
    }
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);

    if(index != -1) {
      bucket = atomic_load(mask_bits + (lid_div_4));

      if(bsid == 0) {
        uint bucket_index = atomic_fetch_add(dst_bucket_counts+bucket, 1u);
        scan_index[get_sub_group_id()][scan_id] = (uint)(bucket_index * num_buckets + bucket);
      }
    }
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);

    if(index != -1) {
      global ulong *bit_string_dst = (global ulong*) (dst_ht + scan_index[get_sub_group_id()][scan_id]);
      bit_string_dst[bsid] = result;
    }

    atomic_init(mask_bits + (lid_div_4), 0);
  }
  return;
}
