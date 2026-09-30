#include "mogg.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static void put16(std::vector<uint8_t> & b, size_t off, uint16_t v) { std::memcpy(b.data()+off, &v, sizeof(v)); }
static void put32(std::vector<uint8_t> & b, size_t off, uint32_t v) { std::memcpy(b.data()+off, &v, sizeof(v)); }
static void put64(std::vector<uint8_t> & b, size_t off, uint64_t v) { std::memcpy(b.data()+off, &v, sizeof(v)); }
static void puti64(std::vector<uint8_t> & b, size_t off, int64_t v) { std::memcpy(b.data()+off, &v, sizeof(v)); }

static std::vector<uint8_t> base_header(size_t size = 64) {
    std::vector<uint8_t> b(size, 0);
    const uint8_t magic[8] = {'M','O','G','G',0,0,0,1};
    std::memcpy(b.data(), magic, sizeof(magic));
    put32(b, 8, 1);   // version
    put32(b, 12, 0);  // metadata count
    put32(b, 16, 0);  // tensor count
    put32(b, 20, 64); // alignment
    put64(b, 24, 40); // metadata offset
    put64(b, 32, 64); // data start
    return b;
}

static bool rejected(const std::vector<uint8_t> & b) {
    try {
        moge::MoggFile f(b.data(), b.size(), false);
        (void) f;
        return false;
    } catch (const std::exception &) {
        return true;
    }
}

int main() {
    auto b = base_header();
    moge::MoggFile owned(b.data(), b.size(), true);
    moge::MoggFile borrowed(b.data(), b.size(), false);
    if (owned.size() != 64 || borrowed.size() != 64) return 2;
    if (!owned.metadata().empty() || !borrowed.tensors().empty()) return 3;
    if (owned.data() == b.data()) return 4;       // copied
    if (borrowed.data() != b.data()) return 5;    // borrowed

    // Hostile counts must fail before reserve()/iteration can allocate based on
    // attacker-controlled values.
    auto huge_meta = b; put32(huge_meta, 12, 0xffffffffu);
    if (!rejected(huge_meta)) return 10;
    auto huge_tensors = b; put32(huge_tensors, 16, 0xffffffffu);
    if (!rejected(huge_tensors)) return 11;

    // Directory parsing is bounded by data_start; a tensor record cannot be
    // smuggled into payload bytes.
    auto no_directory = b; put32(no_directory, 16, 1);
    if (!rejected(no_directory)) return 12;

    // Construct one F32 tensor with a deliberately inconsistent payload size.
    // Header 40 + tensor record 55 bytes rounds data_start to 128.
    std::vector<uint8_t> bad_size(192, 0);
    const uint8_t magic[8] = {'M','O','G','G',0,0,0,1};
    std::memcpy(bad_size.data(), magic, sizeof(magic));
    put32(bad_size, 8, 1); put32(bad_size, 12, 0); put32(bad_size, 16, 1); put32(bad_size, 20, 64);
    put64(bad_size, 24, 40); put64(bad_size, 32, 128);
    size_t p = 40;
    put16(bad_size, p, 1); p += 2; bad_size[p++] = 'x';
    put16(bad_size, p, static_cast<uint16_t>(moge::MoggType::F32)); p += 2;
    bad_size[p++] = 1; bad_size[p++] = 0;
    puti64(bad_size, p, 4); p += 8;
    for (int i=1;i<4;++i) { puti64(bad_size, p, 1); p += 8; }
    put64(bad_size, p, 128); p += 8;
    put64(bad_size, p, 8); p += 8; // should be 16 bytes
    if (!rejected(bad_size)) return 13;

    std::cout << "mogg_memory=ok\n";
    return 0;
}
