
#include <blake2.h>
#include <boost/program_options.hpp>
#include <endian.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <thread>
#include <ratio>
#include <numeric>

#include "blake.c"
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

static const int ZCASH_BLOCK_HEADER_LEN = 140;
static const int N_ZERO_BYTES = 12;
static const int ZCASH_HASH_LEN = 50;

namespace po = boost::program_options;

using std::array;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::nanoseconds;
using std::string;
using std::vector;
using std::minmax_element;
using std::tie;

/* Writes Zcash personalization string. */
static void zcashPerson(uint8_t *person, const int n, const int k) {
  memcpy(person, "ZcashPoW", 8);
  *(uint32_t *)(person + 8) = htole32(n);
  *(uint32_t *)(person + 12) = htole32(k);
}

static void digestInit(blake2b_state *S, const int n, const int k) {
  blake2b_param P[1];

  memset(P, 0, sizeof(blake2b_param));
  P->fanout = 1;
  P->depth = 1;
  P->digest_length = 50; //(512 / n) * n / 8;
  zcashPerson(P->personal, n, k);
  blake2b_init_param(S, P);
}

vector<cl::Device> devices_;
void init_opencl() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  cl::Platform plat;
  for (auto &p : platforms) {
    std::string platname = p.getInfo<CL_PLATFORM_NAME>();
    printf("Platform Name: %s\n", platname.c_str());
    cl::Platform::setDefault(p);
  }

  printf("Getting device\n");
  platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices_);
  printf("Number of devices found: %lu\n", devices_.size());
  for (auto &d : devices_) {
    printf("Device: %s\n", d.getInfo<CL_DEVICE_NAME>().c_str());
  }
  cl::Context ctx(devices_[0]);
  cl::Context::setDefault(ctx);
  cl::Device::setDefault(devices_[0]);

  cl::CommandQueue queue(cl::QueueProperties::Profiling);
  cl::CommandQueue::setDefault(queue);
}

// Loads the OpenCL program
cl::Program load_program(const char *binary_name) {
  std::string name(binary_name);
  std::ifstream source_stream(name, std::ios::binary);
  cl::Program::Sources sources{
      {std::string(std::istreambuf_iterator<char>(source_stream),
                   std::istreambuf_iterator<char>())}};

  std::vector<cl::Device> dev = {cl::Device::getDefault()};
  return cl::Program(sources);
}

void fatal(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  exit(EXIT_FAILURE);
}

uint8_t hex2val(const char *base, size_t off) {
  const char c = base[off];
  if (c >= '0' && c <= '9')
    return c - '0';
  else if (c >= 'a' && c <= 'f')
    return 10 + c - 'a';
  else if (c >= 'A' && c <= 'F')
    return 10 + c - 'A';
  fatal("Invalid hex char at offset %zd: ...%c...\n", off, c);
  return 0;
}
uint32_t parse_header(uint8_t *h, size_t h_len, const char *hex) {
  size_t hex_len;
  size_t bin_len;
  size_t opt0 = ZCASH_BLOCK_HEADER_LEN;
  size_t i;
  if (!hex) {
    return opt0;
  }
  hex_len = strlen(hex);
  bin_len = hex_len / 2;
  if (hex_len % 2)
    fatal("Error: input header must be an even number of hex digits\n");
  if (bin_len != opt0)
    fatal("Error: input header must be a %zd-byte full header\n", opt0);
  assert(bin_len <= h_len);
  for (i = 0; i < bin_len; i++)
    h[i] = hex2val(hex, i * 2) * 16 + hex2val(hex, i * 2 + 1);
  while (--i >= bin_len - N_ZERO_BYTES)
    if (h[i])
      fatal("Error: last %d bytes of full header (ie. last %d "
            "bytes of 32-byte nonce) must be zero due to an "
            "optimization in my BLAKE2b implementation\n",
            N_ZERO_BYTES, N_ZERO_BYTES);
  return bin_len;
}


auto
in_nanoseconds(cl::Event event) {
  auto start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  auto eee = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  return std::chrono::nanoseconds(eee - start);
}

void check_endiness() {
  unsigned int i = 1;
  char *c = (char *)&i;
  if (*c) printf("Little endian\n");
  else printf("Big endian\n");
}

void print_collision_histogram(vector<unsigned> counts, uint buckets) {
  int histcount = std::min(16u,buckets);
  printf("histcount: %d\n", histcount);
  vector<unsigned> hist(histcount, 0);
  int total = 0;
  for(unsigned count : counts) {
    hist[count]++;
    total += count;
  }
  printf("hist:  ");
  for(int h : hist) {
    printf("%d ", h);
  }
  printf(" total: %d\n", total);
}

template<typename T>
void print_stats(cl::Buffer buf) {
  size_t bytes = buf.getInfo<CL_MEM_SIZE>();
  size_t elements = bytes / sizeof(T);
  vector<unsigned> compressed_counts(elements, 0);
  cl::copy(buf, begin(compressed_counts), end(compressed_counts));
  vector<unsigned> counts;
  counts.reserve(elements*8);
  for(auto& cc : compressed_counts) {
    for(int j = 0; j < 8; j++) {
      counts.push_back((cc >> (j * 4)) & 0xf);
    }
  }

  auto result = minmax_element(begin(counts), end(counts));
  printf("min: %u max: %u\n", *result.first, *result.second);

  print_collision_histogram(counts, *result.second + 2);

  printf("counts: \n");
  for (int i = 0; i < 100; i++) {
    printf("%u ", counts[i]);
  }
  printf("\n");
}

void print_info(const char* name, cl::Buffer buf) {
  size_t bytes = buf.getInfo<CL_MEM_SIZE>();
  printf("%s size: %lu MB\n", name, bytes / (1024 * 1024));
}

void print_ht_transposed(cl::Buffer hash_table, int index = 0, int num_entries = 10) {
  size_t elements = hash_table.getInfo<CL_MEM_SIZE>() / sizeof(cl_ulong);
  vector<cl_ulong> table(elements, 0);
  cl::copy(hash_table, begin(table), end(table));

  for (int i = index; i < index + num_entries; i++) {
    printf("[%016lx, %016lx, %016lx, %016lx]    ", table[i * 4],
           table[i * 4 + 1], table[i * 4 + 2], table[i * 4 + 3]);
    int next = i + (1 << 200/(9+1));
    printf("[%016lx, %016lx, %016lx, %016lx]    ", table[next * 4],
           table[next * 4 + 1], table[next * 4 + 2], table[next * 4 + 3]);
    next += (1 << 200/(9+1));
    printf("[%016lx, %016lx, %016lx, %016lx]\n", table[next * 4],
           table[next * 4 + 1], table[next * 4 + 2], table[next * 4 + 3]);
  }
}

void print_ht(cl::Buffer hash_table, int num_bins, int index = 0, int num_entries = 10) {
  size_t elements = hash_table.getInfo<CL_MEM_SIZE>() / sizeof(cl_ulong);
  vector<cl_ulong> table(elements, 0);
  cl::copy(hash_table, begin(table), end(table));

  for (int i = index * num_bins; i < (index + num_entries)* num_bins ; i+=num_bins) {
    printf("[%016lx, %016lx, %016lx, %016lx]  ", table[i * 4],
           table[i * 4 + 1], table[i * 4 + 2], table[i * 4 + 3]);
    int next = i+1;
    printf("[%016lx, %016lx, %016lx, %016lx]  ", table[next * 4],
           table[next * 4 + 1], table[next * 4 + 2], table[next * 4 + 3]);
    next += 1;
    printf("[%016lx, %016lx, %016lx, %016lx]  ", table[next * 4],
           table[next * 4 + 1], table[next * 4 + 2], table[next * 4 + 3]);
    next += 1;
    printf("[%016lx, %016lx, %016lx, %016lx]\n", table[next * 4],
           table[next * 4 + 1], table[next * 4 + 2], table[next * 4 + 3]);
  }
}


int main(int argc, char **argv) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help", "Displays this help message")(
      "bitstring_length,n", po::value<int>()->default_value(200),
      "bit string length")("k_val,k", po::value<int>()->default_value(9),
                           "ZCash k value");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  check_endiness();

  int n = 200, k = 9;

  uint8_t header[ZCASH_BLOCK_HEADER_LEN] = {0};
  uint32_t header_len;
  {
    const uint8_t *input = reinterpret_cast<const uint8_t *>(
        "04000000e54c27544050668f272ec3b460e1cde745c6b21239a81dae637fde47040000"
        "00844bc0c55696ef9920eeda11c1eb41b0c2e7324b46cc2e7aa0c2aa7736448d7a0000"
        "00000000000000000000000000000000000000000000000000000000000068241a587e"
        "7e061d250e00000000000001000000000000000000000000000000000000000000000"
        "0");
    header_len = parse_header(header, sizeof(header), (const char *)input);
  }

  printf("====================\n");

  init_opencl();
  printf("hehehe\n");
  cl::Program blake = load_program("../blake.cl");
  cl::Program equihash = load_program("../equihash.cl");
  try {
    blake.build("-cl-std=CL2.0");
    equihash.build("-cl-std=CL2.1 -O2 -cl-denorms-are-zero -cl-single-precision-constant -cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math");
    auto blake_kern = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int>(blake, "blake");
    auto equihash_kern = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_ulong4, cl_ulong4, cl_int4, unsigned, unsigned>(equihash, "equihash_round");

    blake2b_state_t blake_state;
    zcash_blake2b_init(&blake_state, ZCASH_HASH_LEN, n, k);
    zcash_blake2b_update(&blake_state, header, 128, 0);

    cl::Buffer blake_state_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(blake_state.h), &blake_state.h);

    int num_bitstrings = 1 << (n / (k + 1) + 1);
    int hash_table_collision_depth = 12;
    int hash_table_size = num_bitstrings * hash_table_collision_depth;
    // it is supposed to be 200 but we are storing them in 4 unsigned long
    // long(256bits)
    int bitstring_size = sizeof(cl_ulong4);

    int num_buckets = 1 << n / (k + 1); // 20 bits

    // Hash table buffer. Each bit_string is a ulong4(32 bytes) lower 25(200 bits)
    // are important

    array<cl::Buffer, 2> hash_tables = {
      cl::Buffer(CL_MEM_READ_WRITE, hash_table_size * bitstring_size),
      cl::Buffer(CL_MEM_READ_WRITE, hash_table_size * bitstring_size)
    };
    array<cl::Buffer, 2> bucket_counts = {
      cl::Buffer(CL_MEM_READ_WRITE, num_buckets * sizeof(unsigned int) / 8),
      cl::Buffer(CL_MEM_READ_WRITE, num_buckets * sizeof(unsigned int) / 8)
    };


      //cl::Buffer hash_table(CL_MEM_READ_WRITE, hash_table_size * bitstring_size);
      //cl::Buffer hash_table2(CL_MEM_READ_WRITE, hash_table_size * bitstring_size);
    //cl::Buffer bucket_count(CL_MEM_READ_WRITE, num_buckets * sizeof(unsigned int) / 8);
    //cl::Buffer bucket_count2(CL_MEM_READ_WRITE, num_buckets * sizeof(unsigned int) / 8);
    //cl::Buffer scan_buffer(CL_MEM_READ_WRITE, num_buckets * sizeof(unsigned));

    print_info("hash_table", hash_tables[0]);
    print_info("bucket_count", bucket_counts[0]);
    print_info("hash_table2", hash_tables[1]);
    print_info("bucket_count2", bucket_counts[1]);

    int pattern = 0;
    for(auto &ht : hash_tables) {
      cl::CommandQueue::getDefault().enqueueFillBuffer(ht, 0ULL, 0, hash_table_size * bitstring_size);
    }
    for(auto &bc : bucket_counts) {
      cl::CommandQueue::getDefault().enqueueFillBuffer(bc, pattern, 0, num_buckets * sizeof(unsigned int)/8);
    }

    printf("Running blake2\n");
    auto blake_event = blake_kern(cl::EnqueueArgs(cl::NDRange(num_bitstrings / 2)),
                                  blake_state_buf, hash_tables[0], bucket_counts[0], hash_table_collision_depth);
    print_ht(hash_tables[0], hash_table_collision_depth);
    print_stats<unsigned>(bucket_counts[0]);

    vector<cl_ulong4> masks = {
      { 0xfffff00000000000, 0, 0, 0 },
      { 0x00000fffff000000, 0, 0, 0 },
      { 0x0000000000fffff0, 0, 0, 0 },
      { 0x000000000000000f, 0xffff000000000000, 0, 0 },
      { 0                 , 0x0000fffff0000000, 0, 0 },
      { 0                 , 0x000000000fffff00, 0, 0 }
    };

    vector<cl_int4> mask_shifts = {
      { 0, 0, 0, 0 },
      { 24, 0, 0, 0 },
      { 4, 0, 0, 0 },
      { -16, 64-16, 0, 0 },
      { 0, 64-36, 0, 0 },
      { 0, 64-56, 0, 0 }
    };

    printf("Running equihash\n");
    for(int i = 0; i < 1; i++) {
      printf("round: %d\n", i);
      cl_ulong4 mask = masks[i+1];
      cl_int4 mask_shift = mask_shifts[i+1];
      auto equihash_event2 = equihash_kern(cl::EnqueueArgs(cl::NDRange((1<<20) * 4), cl::NDRange(128)),
                                           hash_tables[1], bucket_counts[1],
                                           hash_tables[0], bucket_counts[0],
                                           masks[i], mask, mask_shift, num_buckets, 2);

      cl::CommandQueue::getDefault().enqueueFillBuffer(hash_tables[0], 0ULL, 0, hash_table_size * bitstring_size);
      cl::CommandQueue::getDefault().enqueueFillBuffer(bucket_counts[0], pattern, 0, num_buckets * sizeof(unsigned int)/8);
      print_ht(hash_tables[1], hash_table_collision_depth);
      print_stats<unsigned>(bucket_counts[1]);
      std::swap(hash_tables[0], hash_tables[1]);
      std::swap(bucket_counts[0], bucket_counts[1]);
    }


    cl::CommandQueue::getDefault().flush();
    cl::CommandQueue::getDefault().finish();
    printf("Blake Time: %lu us\n", duration_cast<microseconds>(in_nanoseconds(blake_event)).count());
    // printf("Equihash 2: %lu us\n", duration_cast<microseconds>(in_nanoseconds(equihash_event2)).count());
    // printf("Equihash 3: %lu us\n", duration_cast<microseconds>(in_nanoseconds(equihash_event3)).count());
    // printf("Equihash 4: %lu us\n", duration_cast<microseconds>(in_nanoseconds(equihash_event4)).count());
    // printf("Equihash 5: %lu us\n", duration_cast<microseconds>(in_nanoseconds(equihash_event5)).count());
    // printf("Equihash 6: %lu us\n", duration_cast<microseconds>(in_nanoseconds(equihash_event6)).count());


  } catch(cl::Error &err) {

    if(err.err() == CL_BUILD_PROGRAM_FAILURE) {
      printf("Build Failed\n");
      string log;
      blake.getBuildInfo(cl::Device::getDefault(), CL_PROGRAM_BUILD_LOG, &log);
      printf("Blake LOG: %s", log.c_str());
      equihash.getBuildInfo(cl::Device::getDefault(), CL_PROGRAM_BUILD_LOG, &log);
      printf("Equihash LOG: %s", log.c_str());
    }
    printf("OpenCL Error: %s %d", err.what(), err.err());
    return EXIT_FAILURE;
  }
}
